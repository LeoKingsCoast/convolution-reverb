#include <portaudio.h>
#include <stdio.h>

PaStream* portaudio_init(int sample_rate, PaSampleFormat sample_type, int frames_per_buffer, PaStreamCallback callback, void* user_data) {
    if (sample_rate < 0 || frames_per_buffer < 0) {
        fprintf(stderr, "[Error]: Sample Rate and Frames per Buffer must be positive\n");
        return NULL;
    }

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to initialize portaudio: %s\n", Pa_GetErrorText(err));
        return NULL;
    }

    PaStreamParameters in_params;
    in_params.device = Pa_GetDefaultInputDevice();
    if (in_params.device == paNoDevice) {
        fprintf(stderr, "[Error]: No input device found\n");
        return NULL;
    }
    in_params.channelCount = 1;
    in_params.sampleFormat = sample_type;
    in_params.suggestedLatency = Pa_GetDeviceInfo(in_params.device)->defaultLowInputLatency;
    in_params.hostApiSpecificStreamInfo = NULL;

    PaStreamParameters out_params;
    out_params.device = Pa_GetDefaultOutputDevice();
    if (out_params.device == paNoDevice) {
        fprintf(stderr, "[Error]: No output device found\n");
        return NULL;
    }
    out_params.channelCount = 2;
    out_params.sampleFormat = sample_type;
    out_params.suggestedLatency = Pa_GetDeviceInfo(out_params.device)->defaultLowOutputLatency;
    out_params.hostApiSpecificStreamInfo = NULL;

    PaStream* stream;
    err = Pa_OpenStream(
        &stream, &in_params, &out_params, sample_rate, frames_per_buffer, 0,
        callback, user_data
    );

    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to open stream: %s\n", Pa_GetErrorText(err));
        return NULL;
    }

    return stream;
}
