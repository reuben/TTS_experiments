'''
BSD 3-Clause License

Copyright (c) 2019, Mozilla Corporation
Copyright (c) 2018, NVIDIA Corporation
Copyright (c) 2017, Prem Seetharaman


* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import numpy as np
import torch.nn.functional as F
import os
import scipy

from torch.autograd import Variable
from librosa.util import pad_center, tiny

def window_sumsquare(n_frames, hop_length=200, win_length=800, n_fft=800):
    # type: (int, int, int, int) -> Tensor
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = torch.zeros(n).float()

    # Compute the squared window at the desired length
    win_sq = torch.hann_window(win_length, periodic=True).float()
    lpad = (n_fft - win_length) // 2
    rpad = n_fft - win_length - lpad
    win_sq = torch.nn.functional.pad(win_sq, [lpad, rpad])
    # win_sq = pad_center(win_sq, n_fft)
    # win_sq = torch.from_numpy(win_sq).float()

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.jit.ScriptModule):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    __constants__ = ['filter_length', 'hop_length', 'win_length', 'tiny_fp32']

    def __init__(self, filter_length=800, hop_length=200, win_length=800):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.tiny_fp32 = float(tiny(np.array([1.0], dtype=np.float32)))
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        # assert(win_length >= filter_length)
        # get window and zero center pad it to filter_length
        fft_window = torch.hann_window(win_length, periodic=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        if 'PYTORCH_JIT' not in os.environ or os.environ['PYTORCH_JIT'] == '1':
            self.forward_basis = torch.jit.Attribute(forward_basis.float(), torch.Tensor)
            self.inverse_basis = torch.jit.Attribute(inverse_basis.float(), torch.Tensor)
        else:
            self.register_buffer('forward_basis', forward_basis.float())
            self.register_buffer('inverse_basis', inverse_basis.float())

    @torch.jit.script_method
    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase

    @torch.jit.script_method
    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        window_sum = window_sumsquare(magnitude.size(-1), hop_length=self.hop_length,
            win_length=self.win_length, n_fft=self.filter_length)
        # remove modulation effects
        approx_nonzero_indices = torch.nonzero(window_sum > self.tiny_fp32)[0]
        inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

        # scale by hop ratio
        inverse_transform *= float(self.filter_length) / float(self.hop_length)

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction