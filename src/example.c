#include "convolution.h"

#include <stdio.h>
#include <string.h>
#include <fftw3.h>

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

#define IMPULSE_SIZE 3
#define SIGNAL_SIZE 5

int main() {
    float signal[SIGNAL_SIZE] = {-1,0,1,3,2};
    float impulse_response[IMPULSE_SIZE] = {1,1,1};

    ConvolutorCircular* conv = conv_init_circular(SIGNAL_SIZE, IMPULSE_SIZE, impulse_response);

    memcpy(conv->input, signal, SIGNAL_SIZE * sizeof(float));

    printf("Input: ");
    print_signal(conv->input, conv->input_size);
    printf("\nImpulse response: ");
    print_signal(impulse_response, IMPULSE_SIZE);

    conv_circular(conv);

    printf("\nConvolution: ");
    print_signal(conv->output, conv->input_size);

    conv_terminate_circular(conv);

    return 0;
}
