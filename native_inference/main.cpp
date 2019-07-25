#include <cctype>
#include <cstdio>
#include <codecvt>
#include <locale>
#include <memory>
#include <string>

#include <sys/time.h>
#include <torch/script.h>
#include <unistd.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

using namespace std;

template<typename T>
decltype(auto)
timeit(T a)
{
  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  auto ret = a();
  gettimeofday(&tv2, NULL);
  fprintf(stderr, "Total time = %f seconds\n",
    (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 +
    (double)(tv2.tv_sec - tv1.tv_sec));
  return ret;
}


auto
converter()
{
  return wstring_convert<codecvt_utf8<char32_t>, char32_t>{};
}


string
text_to_phonemes(const string& input, const unordered_set<char32_t>& punct)
{
  string phonemes;
  {
    const char* input_ptr = input.c_str();
    const char* phonemes_c = espeak_TextToPhonemes((const void**)&input_ptr, espeakCHARS_AUTO, 2);
    phonemes = phonemes_c;
    free(const_cast<char*>(phonemes_c));
  }

  // espeak-ng puts a space in the beginning for some reason
  if (phonemes[0] == ' ') {
    phonemes.erase(phonemes.begin());
  }

  // convert input to graphemes, check if last grapheme is punctuation, if so add the (UTF-8 converted) punctuation to the phonemes output
  // this is done because espeak removes punctuation from the input but it is needed by the model
  u32string input_graphemes = converter().from_bytes(input);
  if (punct.count(input_graphemes.back())) {
    phonemes.append(converter().to_bytes(input_graphemes.back()));
  }

  return phonemes;
}


vector<int>
phonemes_to_sequence(const string& phonemes, const unordered_map<char32_t, int> id_map)
{
  vector<int> result;

  for (char32_t phoneme : converter().from_bytes(phonemes)) {
    // ignore stress symbols ˌ and ˈ
    if ((phoneme == U'ˌ') || (phoneme == U'ˈ')) {
      continue;
    }
    auto it = id_map.find(phoneme);
    if (it != id_map.end()) {
      result.push_back(it->second);
    } else {
      string phoneme_bytes = converter().to_bytes(phoneme);
      fprintf(stderr, "Unknown phoneme: \"%s\" (", phoneme_bytes.c_str());
      for (unsigned char c : phoneme_bytes) {
        fprintf(stderr, "%X ", c);
      }
      fprintf(stderr, ")\n");
      exit(1);
    }
  }

  return result;
}


int
main(int argc, char** argv)
{
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <model file> <text to synthesize> <output file (raw signed 16 bit WAV samples)>\n", argv[0]);
    return 1;
  }

  const unordered_set<char32_t> phoneme_punctuation{{U'.', U'!', U';', U':', U',', U'?'}};
  const unordered_map<char32_t, int> phonemes_id_map{{
    make_pair(U'_', 0),
    make_pair(U'~', 1),
    make_pair(U'^', 2),
    make_pair(U'a', 4),
    make_pair(U'b', 5),
    make_pair(U'd', 6),
    make_pair(U'e', 7),
    make_pair(U'f', 8),
    make_pair(U'h', 9),
    make_pair(U'i', 10),
    make_pair(U'j', 11),
    make_pair(U'k', 12),
    make_pair(U'l', 13),
    make_pair(U'm', 14),
    make_pair(U'n', 15),
    make_pair(U'o', 16),
    make_pair(U'p', 17),
    make_pair(U'r', 18),
    make_pair(U's', 19),
    make_pair(U't', 20),
    make_pair(U'u', 21),
    make_pair(U'v', 22),
    make_pair(U'w', 23),
    make_pair(U'x', 24),
    make_pair(U'z', 25),
    make_pair(U'æ', 26),
    make_pair(U'ð', 27),
    make_pair(U'ŋ', 28),
    make_pair(U'ɐ', 29),
    make_pair(U'ɑ', 30),
    make_pair(U'ɒ', 31),
    make_pair(U'ɔ', 32),
    make_pair(U'ə', 33),
    make_pair(U'ɚ', 34),
    make_pair(U'ɛ', 35),
    make_pair(U'ɜ', 36),
    make_pair(U'ɡ', 37),
    make_pair(U'ɪ', 38),
    make_pair(U'ɫ', 39),
    make_pair(U'ɬ', 40),
    make_pair(U'ɹ', 41),
    make_pair(U'ɾ', 42),
    make_pair(U'ʃ', 43),
    make_pair(U'ʊ', 44),
    make_pair(U'ʌ', 45),
    make_pair(U'ʒ', 46),
    make_pair(U'ʔ', 47),
    make_pair(U'ː', 48),
    make_pair(U'θ', 49),
    make_pair(U'ᵻ', 50),
    make_pair(U'!', 51),
    make_pair(U'\'', 52),
    make_pair(U'(', 53),
    make_pair(U')', 54),
    make_pair(U',', 55),
    make_pair(U'-', 56),
    make_pair(U'.', 57),
    make_pair(U':', 58),
    make_pair(U';', 59),
    make_pair(U'?', 60),
    make_pair(U' ', 61)
  }};

  const char* model_file = argv[1];
  string input_text = argv[2];
  const char* output_file = argv[3];

  fprintf(stderr, "Initializing espeak-ng... ");
  int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
  if (sr == -1) {
    fprintf(stderr, "Error initializing espeak-ng\n");
    return 1;
  }
  espeak_VOICE voice_spec;
  voice_spec.name = nullptr; // not specified
  voice_spec.languages = "en-us";
  voice_spec.gender = 0; // not specified
  voice_spec.age = 0; // not specified
  voice_spec.variant = 0; // best match
  espeak_ERROR err = espeak_SetVoiceByProperties(&voice_spec);
  if (err != EE_OK) {
    fprintf(stderr, "Error setting en-us voice\n");
    return 1;
  } else {
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "Loading model... ");
  torch::jit::script::Module module = timeit([&]{ return torch::jit::script::Module(torch::jit::load(model_file)); });

  vector<torch::jit::IValue> inputs;
  fprintf(stderr, "Input text: \"%s\"\n", input_text.c_str());
  string phons = text_to_phonemes(input_text, phoneme_punctuation);
  fprintf(stderr, "Phonemes: \"%s\"\n", phons.c_str());
  vector<int> seq = phonemes_to_sequence(phons, phonemes_id_map);
  // int data[] = {5, 38, 13, 61, 41, 10, 48, 13, 10, 61,  9, 26,  6, 38, 20, 61, 32, 48, 13, 57}; // Bill Rielly had it all.
  inputs.push_back(torch::from_blob(seq.data(), {static_cast<long long>(seq.size())}, at::kInt).toType(at::kLong).unsqueeze(0));

  // Warm up
  fprintf(stderr, "Warming up... ");
  timeit([&]{ return module.get_method("inference")({torch::ones({1, 30}).toType(at::kLong)}); });

  // usleep(1000*1000);

  fprintf(stderr, "Synthesizing... ");
  auto output = timeit([&] { return module.get_method("inference")(inputs); });

  at::Tensor waveform = output.toTuple()->elements()[0].toTensor()
                                                       .squeeze(0)
                                                       .toType(at::kShort);

  float audio_length_s = (float)waveform.sizes()[0] / 22050.f;
  fprintf(stderr, "Synthesized audio duration = %f seconds\n", audio_length_s);

  int16_t* samples = waveform.data<int16_t>();
  FILE* fout = fopen(output_file, "wb");
  fwrite(samples, sizeof(int16_t), waveform.sizes()[0], fout);
  fclose(fout);
  return 0;
}
