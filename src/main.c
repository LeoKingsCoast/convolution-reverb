#include "convolution.h"
#include "audio.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <portaudio.h>

#define IMPULSE_RESP_SIZE 128
#define FRAMES_PER_BUFFER 512
#define SAMPLE_RATE 44100

void print_signal(float* signal, int size) {
    printf("{ %lf", signal[0]);
    for (int i = 1; i < size; i++) {
        printf(", %lf", signal[i]);
    }
    printf(" }");
}

void print_spectrum(fftwf_complex* signal, int size) {
    printf("{\n %lf + i(%lf)", signal[0][0], signal[0][1]);
    for (int i = 1; i < size; i++) {
        printf("\n %lf + i(%lf)", signal[i][0], signal[i][1]);
    }
    printf("\n}");
}

static int callback(
    const void *in_buffer, void *out_buffer, unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info, PaStreamCallbackFlags status_flags,
    void *convolutor
)
{
    const float* in = (const float*) in_buffer;
    float* out = (float*) out_buffer;
    Convolutor* conv = (Convolutor*) convolutor;

    if (!in || !out)
        return paContinue;

    memcpy(conv->input, in, frames_per_buffer * sizeof(float));

    conv_convolve(conv);

    for (int i = 0; i < frames_per_buffer; i++) {
        float sample = conv->output[i];
        *out++ = sample; // left side
        *out++ = sample; // right side
    }

    return paContinue;
}

int main() {
    float impulse_resp[IMPULSE_RESP_SIZE];
    for (int i = 0; i < IMPULSE_RESP_SIZE; i++) {
        impulse_resp[i] = 0.5 * exp(-1.0 * i / IMPULSE_RESP_SIZE);
    }

    Convolutor* conv = conv_init(FRAMES_PER_BUFFER, IMPULSE_RESP_SIZE, impulse_resp);

    print_spectrum(conv->impulse_freq_padded, (conv->window_size / 2 + 1));

    PaStream* stream = portaudio_init(SAMPLE_RATE, paFloat32, FRAMES_PER_BUFFER, callback, conv);
    if (!stream) {
        return -1;
    }

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to start stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }

    printf("Press Enter to stop the stream\n");
    getchar();

    conv_terminate(conv);

    Pa_CloseStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to close stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }
    Pa_Terminate();

    return 0;
}
