#include <stdio.h>
#include <fftw3.h>
#include "convolution.h"

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

#define IMPULSE_SIZE 3
#define SIGNAL_SIZE 9

int main() {
    // double signal[SIGNAL_SIZE] = {8,6,-3,4,1,-6,7,9,-1};
    double impulse_response[IMPULSE_SIZE] = {1,5,4};

    Convolutor* conv = conv_init(SIGNAL_SIZE, IMPULSE_SIZE, impulse_response);

    print_spectrum(conv->out_freq_buffer, conv->window_size / 2 + 1);

    conv_terminate(conv);

    return 0;
}
