import torch
import librosa
import numpy as np
import os
from scipy import signal, io

from torch import nn
from .generic_utils import load_config
from .stft import STFT


class InvMelSpectrogram(torch.jit.ScriptModule):
    __constants__ = ['griffin_lim_iters', 'preemphasis', 'sample_rate',
                     'do_trim_silence', 'signal_norm', 'symmetric_norm',
                     'clip_norm', 'max_norm', 'min_level_db', 'ref_level_db',
                     'power']

    def __init__(self,
                 tts,
                 trace_inputs,
                 bits=None,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=None,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 **kwargs):
        super(InvMelSpectrogram, self).__init__()

        self.tts = torch.jit.trace_module(tts, trace_inputs)
        self.bits = bits
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = 0 if mel_fmin is None else mel_fmin
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        self.stft = STFT(self.n_fft, self.hop_length, self.win_length)
        mel_basis = self._build_mel_basis()
        inv_mel_basis = np.linalg.pinv(mel_basis)
        if 'PYTORCH_JIT' not in os.environ or os.environ['PYTORCH_JIT'] == '1':
            self._mel_basis = torch.jit.Attribute(torch.from_numpy(mel_basis).float(), torch.Tensor)
            self._inv_mel_basis = torch.jit.Attribute(torch.from_numpy(inv_mel_basis).float(), torch.Tensor)
        else:
            self._mel_basis = torch.from_numpy(mel_basis).float()
            self._inv_mel_basis = torch.from_numpy(inv_mel_basis).float()

        members = vars(self)
        for key, value in members.items():
            if 'basis' not in key:
                print(" | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    @torch.jit.script_method
    def _mel_to_linear(self, mel_spec):
        dot = torch.matmul(self._inv_mel_basis, mel_spec)
        return torch.max(torch.tensor(1e-10, dtype=torch.float32), dot)

    def _build_mel_basis(self, ):
        n_fft = (self.num_freq - 1) * 2
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _denormalize(self, S):
        """denormalize values"""
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = S_denorm.clamp(-self.max_norm, self.max_norm)
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = S_denorm.clamp(0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db / self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        return n_fft, hop_length, win_length

    def _db_to_amp(self, x):
        # type: (Tensor) -> Tensor
        return torch.pow(torch.tensor(10.0, dtype=torch.float), x.float() * 0.05)

    # TODO: convert to PyTorch (JIT)
    # def apply_inv_preemphasis(self, x):
    #     if self.preemphasis == 0:
    #         raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
    #     return signal.lfilter([1], [1, -self.preemphasis], x)

    @torch.jit.script_method
    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        # if self.preemphasis != 0:
        #     return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S.unsqueeze(0)**self.power).squeeze(0)

    @torch.jit.script_method
    def inference(self, text):
        decoder_output, postnet_output, alignments, stop_tokens = self.tts.inference(text)
        postnet_output = postnet_output[0].t()
        alignment = alignments[0]
        decoder_output = decoder_output[0]
        wav = self.inv_mel_spectrogram(postnet_output)
        if self.do_trim_silence:
            wav = wav[:self.find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8)]
        # normalize samples
        wav_norm = wav * float(32767. / float(torch.max(torch.tensor(0.01, dtype=torch.float32), torch.max(torch.abs(wav)))))
        wav_norm = wav_norm.to(torch.int16)
        return wav_norm, alignment, decoder_output, postnet_output, stop_tokens

    @torch.jit.script_method
    def _griffin_lim(self, S):
        # original numpy:
        # angles = np.angle(np.exp(2j * np.pi * np.random.rand(*S.size())))
        #
        # expand complex math by hand to run in pytorch JIT:
        # np.angle(x) = atan2(x.imag, x.real)
        #
        # 2j = (0, 2)
        # np.pi = (np.pi, 0)
        # np.random.rand(*S.size()) = randS = (randS, 0)
        #
        # simplifying:
        #
        # (0, 2i) * (pi, 0) = (0, 2 * pi)
        # (0, 2pi) * (randS) = (0, 2 * pi * randS)
        #
        # final pytorch code:
        # angles = atan2(2 * pi * randS, 0)
        angles = torch.atan2(2 * np.pi * torch.rand_like(S), torch.zeros(1))
        signal = self.stft.inverse(S, angles).squeeze(1)

        for i in range(self.griffin_lim_iters):
            _, angles = self.stft.transform(signal)
            signal = self.stft.inverse(S, angles).squeeze(1)
        return signal

    @torch.jit.script_method
    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        # type: (Tensor, int, float) -> int
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(torch.tensor(threshold_db, dtype=torch.float))
        ret = len(wav)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if torch.max(wav[x:x + window_length]) < threshold:
                ret = x + hop_length
                break
        return ret
