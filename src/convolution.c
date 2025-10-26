#include "convolution.h"

#include <fftw3.h>
#include <stdlib.h>
#include <string.h>

Convolutor* conv_init(int input_size, int impulse_resp_size, double* impulse_resp) {
    if (input_size <= 0 || impulse_resp_size <= 0 || impulse_resp == NULL)
        return NULL;

    Convolutor* c = (Convolutor*) malloc(sizeof(Convolutor));
    if (!c)
        return NULL;

    c->input_size = input_size;
    c->impulse_resp_size = impulse_resp_size;
    c->window_size = 2 * c->impulse_resp_size - 1;
    c->overlap_size = c->impulse_resp_size - 1;
    c->output_size = c->input_size + c->impulse_resp_size - 1;

    c->input = malloc(c->input_size * sizeof(double));
    c->impulse_resp = malloc(c->impulse_resp_size * sizeof(double));
    c->window = malloc(c->window_size * sizeof(double));
    c->overlap = malloc(c->overlap_size * sizeof(double));
    c->out_block = malloc(c->window_size * sizeof(double));
    c->output = malloc(c->output_size * sizeof(double));

    c->window_freq = fftw_malloc((c->window_size / 2 + 1) * sizeof(fftw_complex));
    c->impulse_freq_padded = fftw_malloc((c->window_size / 2 + 1) * sizeof(fftw_complex));
    c->out_block_freq = fftw_malloc((c->window_size / 2 + 1) * sizeof(fftw_complex));

    c->in_time_buffer = malloc(c->window_size * sizeof(double));
    c->in_freq_buffer = fftw_malloc((c->window_size / 2 + 1) * sizeof(fftw_complex));
    c->out_time_buffer = malloc(c->window_size * sizeof(double));
    c->out_freq_buffer = fftw_malloc((c->window_size / 2 + 1) * sizeof(fftw_complex));
    c->fft_plan = fftw_plan_dft_r2c_1d(c->window_size, c->in_time_buffer, c->out_freq_buffer, FFTW_MEASURE);
    c->ifft_plan = fftw_plan_dft_c2r_1d(c->window_size, c->in_freq_buffer, c->out_time_buffer, FFTW_MEASURE);

    if (!c->input || !c->window || !c->overlap || !c->out_block || !c->output ||
        !c->window_freq || !c->impulse_freq_padded || !c->out_block_freq ||
        !c->in_time_buffer || !c->in_freq_buffer || !c->out_time_buffer ||
        !c->out_freq_buffer || !c->fft_plan || !c->ifft_plan) 
    {
        free(c);
        return NULL;
    }

    memcpy(c->impulse_resp, impulse_resp, c->impulse_resp_size * sizeof(double));
    memcpy(c->in_time_buffer, c->impulse_resp, c->impulse_resp_size * sizeof(double));

    // Copy the impulse response to the fft input, padding with 0 until the window size
    for (int i = 0; i < c->window_size; i++) {
        if (i < c->impulse_resp_size) c->in_time_buffer[i] = c->impulse_resp[i];
        else c->in_time_buffer[i] = 0;
    }

    // impulse response DFT stored in out_freq_buffer
    fftw_execute(c->fft_plan);
    memcpy(c->impulse_freq_padded, c->out_freq_buffer, (c->window_size / 2 + 1) * sizeof(fftw_complex));

    return c;
}

void conv_terminate(Convolutor* c) {
    if (c) {
        fftw_destroy_plan(c->ifft_plan);
        fftw_destroy_plan(c->fft_plan);
        fftw_free(c->out_freq_buffer);
        free(c->out_time_buffer);
        fftw_free(c->in_freq_buffer);
        free(c->in_time_buffer);
        fftw_free(c->window_freq);
        fftw_free(c->impulse_freq_padded);
        fftw_free(c->out_block_freq);
        free(c->input);
        free(c->impulse_resp);
        free(c->window);
        free(c->overlap);
        free(c->out_block);
        free(c->output);

        free(c);
    }
}
