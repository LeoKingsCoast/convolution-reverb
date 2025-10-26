#include <portaudio.h>

PaStream* portaudio_init(int sample_rate, PaSampleFormat sample_type, int frames_per_buffer, PaStreamCallback callback);
