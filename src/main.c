#include "convolution.h"
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <portaudio.h>

#define IMPULSE_RESP_SIZE 256
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

typedef struct user_data {
    ConvolutorCircular* convolutor;
    float* overlap_buffer;
    int impulse_resp_size;
} UserData;

static int callback(
    const void *in_buffer, void *out_buffer, unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info, PaStreamCallbackFlags status_flags,
    void *user_data
)
{
    const float* in = (const float*) in_buffer;
    float* out = (float*) out_buffer;
    UserData* params = (UserData*) user_data;

    ConvolutorCircular* conv = params->convolutor;
    float* overlap_buffer = params->overlap_buffer;
    int impulse_resp_size = params->impulse_resp_size;

    if (!in || !out)
        return paContinue;

    memcpy(conv->input, overlap_buffer, (impulse_resp_size - 1) * sizeof(float));
    memcpy(conv->input + (impulse_resp_size - 1), in, frames_per_buffer * sizeof(float));

    conv_circular(conv);

    for (int i = 0; i < frames_per_buffer; i++) {
        float sample = conv->output[impulse_resp_size - 1 + i];
        *out++ = sample; // left side
        *out++ = sample; // right side
    }

    for (int i = 0; i < impulse_resp_size - 1; i++) {
        overlap_buffer[i] = in[frames_per_buffer - impulse_resp_size + i];
    }

    return paContinue;
}

int main() {
    float impulse_resp[IMPULSE_RESP_SIZE];

    float decay_time = 5.0;    // seconds until ~-60dB (approx)

    for (int i = 0; i < IMPULSE_RESP_SIZE; i++) {
        float t = i / (float) SAMPLE_RATE;
        impulse_resp[i] = 0.08 * exp(-t / decay_time);
    }

    UserData user_data = {
        .convolutor = conv_init_circular(FRAMES_PER_BUFFER + IMPULSE_RESP_SIZE - 1, IMPULSE_RESP_SIZE, impulse_resp),
        .overlap_buffer = calloc((IMPULSE_RESP_SIZE - 1), sizeof(float)),
        .impulse_resp_size = IMPULSE_RESP_SIZE,
    };

    PaStream* stream = portaudio_init( SAMPLE_RATE, paFloat32, FRAMES_PER_BUFFER, callback, &user_data);
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

    free(user_data.overlap_buffer);
    conv_terminate_circular(user_data.convolutor);

    Pa_CloseStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to close stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }
    Pa_Terminate();

    return 0;
}
