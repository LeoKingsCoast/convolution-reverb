#include "convolution.h"
#include "audio.h"

#include <stdio.h>
#include <string.h>
#include <fftw3.h>
#include <portaudio.h>

void print_signal(double* signal, int size) {
    printf("{ %lf", signal[0]);
    for (int i = 1; i < size; i++) {
        printf(", %lf", signal[i]);
    }
    printf(" }");
}

void print_spectrum(fftw_complex* signal, int size) {
    printf("{\n %lf + i(%lf)", signal[0][0], signal[0][1]);
    for (int i = 1; i < size; i++) {
        printf("\n %lf + i(%lf)", signal[i][0], signal[i][1]);
    }
    printf("\n}");
}

static int callback(
    const void *in_buffer, void *out_buffer, unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info, PaStreamCallbackFlags status_flags,
    void *user_data
)
{
    const double* in = (const double*) in_buffer;
    double* out = (double*) out_buffer;

    if (!in || !out)
        return paContinue;

    for (int i = 0; i < frames_per_buffer; i++) {
        double sample = *in++;
        *out++ = sample; // left side
        *out++ = sample; // right side
    }

    return paContinue;
}

int main() {
    PaStream* stream = portaudio_init(44100, paFloat32, 64, callback);
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

    Pa_CloseStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to close stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }
    Pa_Terminate();

    return 0;
}
