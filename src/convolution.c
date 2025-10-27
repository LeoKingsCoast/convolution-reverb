/***************************************************************************
 *
 * @file convolution.c
 * @brief Implementation of the Convolutor object, meant to be used in real-time
 * signal processing applications. It allocates all buffers and FFT plans at
 * initialization, leaving minimal overhead for computing the convolution.
 *
 * @author Leonardo Costa
 * @date 2025-10-26
 *
 **************************************************************************/

#include "convolution.h"

#include <fftw3.h>
#include <stdlib.h>
#include <string.h>

Convolutor* conv_init(int input_size, int impulse_resp_size, float* impulse_resp) {
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

    c->input = malloc(c->input_size * sizeof(float));
    c->impulse_resp = malloc(c->impulse_resp_size * sizeof(float));
    c->window = malloc(c->window_size * sizeof(float));
    c->overlap = malloc(c->overlap_size * sizeof(float));
    c->out_block = malloc(c->window_size * sizeof(float));
    c->output = malloc(c->output_size * sizeof(float));

    c->window_freq = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->impulse_freq_padded = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->out_block_freq = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));

    c->in_time_buffer = malloc(c->window_size * sizeof(float));
    c->in_freq_buffer = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->out_time_buffer = malloc(c->window_size * sizeof(float));
    c->out_freq_buffer = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->fft_plan = fftwf_plan_dft_r2c_1d(c->window_size, c->in_time_buffer, c->out_freq_buffer, FFTW_MEASURE);
    c->ifft_plan = fftwf_plan_dft_c2r_1d(c->window_size, c->in_freq_buffer, c->out_time_buffer, FFTW_MEASURE);

    if (!c->input || !c->window || !c->overlap || !c->out_block || !c->output ||
        !c->window_freq || !c->impulse_freq_padded || !c->out_block_freq ||
        !c->in_time_buffer || !c->in_freq_buffer || !c->out_time_buffer ||
        !c->out_freq_buffer || !c->fft_plan || !c->ifft_plan) 
    {
        free(c);
        return NULL;
    }

    memcpy(c->impulse_resp, impulse_resp, c->impulse_resp_size * sizeof(float));
    memcpy(c->in_time_buffer, c->impulse_resp, c->impulse_resp_size * sizeof(float));

    // Copy the impulse response to the fft input, padding with 0 until the window size
    for (int i = 0; i < c->window_size; i++) {
        if (i < c->impulse_resp_size) c->in_time_buffer[i] = c->impulse_resp[i];
        else c->in_time_buffer[i] = 0;
    }

    // impulse response DFT stored in out_freq_buffer
    fftwf_execute(c->fft_plan);
    memcpy(c->impulse_freq_padded, c->out_freq_buffer, (c->window_size / 2 + 1) * sizeof(fftwf_complex));

    return c;
}

void conv_terminate(Convolutor* c) {
    if (c) {
        fftwf_destroy_plan(c->ifft_plan);
        fftwf_destroy_plan(c->fft_plan);
        fftwf_free(c->out_freq_buffer);
        free(c->out_time_buffer);
        fftwf_free(c->in_freq_buffer);
        free(c->in_time_buffer);
        fftwf_free(c->window_freq);
        fftwf_free(c->impulse_freq_padded);
        fftwf_free(c->out_block_freq);
        free(c->input);
        free(c->impulse_resp);
        free(c->window);
        free(c->overlap);
        free(c->out_block);
        free(c->output);

        free(c);
    }
}

void conv_convolve(Convolutor* c) {
    // After surpassing the input size, in_pos continues so that it stores the last block in the output
    for (int in_pos = 0, window_pos = 0; in_pos < c->input_size + c->window_size; in_pos++, window_pos++) {
        // If the FFT input is filled
        if (window_pos == c->impulse_resp_size && in_pos < (c->input_size + c->impulse_resp_size)) {
            /* 
             * Store overlap of the last window convolution. Before the first
             * convolution occurs it won't matter because the output write 
             * operations will skip it.
             */
            for (int i = 0; i < c->overlap_size; i++) {
                c->overlap[i] = c->out_time_buffer[c->impulse_resp_size + i] / c->window_size;
            }

            // Pad buffer with zeros, the FFT input before this is already filled
            for (int i = c->impulse_resp_size; i < c->window_size; i++)
                c->in_time_buffer[i] = 0;
            
            // Store DFT of this window in out_freq_buffer
            fftwf_execute(c->fft_plan);

            /*
             * Multiply window frequency signal by the impulse response spectrum
             * Complex Multiplication: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
             */
            for (int i = 0; i < (c->window_size / 2 + 1); i++) {
                c->in_freq_buffer[i][0] = c->out_freq_buffer[i][0] * c->impulse_freq_padded[i][0]
                                        - c->out_freq_buffer[i][1] * c->impulse_freq_padded[i][1];
                c->in_freq_buffer[i][1] = c->out_freq_buffer[i][0] * c->impulse_freq_padded[i][1]
                                        + c->out_freq_buffer[i][1] * c->impulse_freq_padded[i][0];
            }

            /*
             * Go back to time domain
             * Store this window's convolution result in out_time_buffer
             * We need to normalize the IFFT result every time we use it (divide
             * by window_size)
             */
            fftwf_execute(c->ifft_plan);

            window_pos = 0;
        }

        // Fill FFT input buffer with the input signal
        if (in_pos < c->input_size)
            c->in_time_buffer[window_pos] = c->input[in_pos];
        else
            c->in_time_buffer[window_pos] = 0;

        // Only start filling the output after the first convolution
        // The output is always one output block late relative to the input
        if (in_pos >= c->impulse_resp_size) {
            if (window_pos < c->overlap_size)
                c->output[in_pos - c->impulse_resp_size] = c->out_time_buffer[window_pos] / c->window_size + c->overlap[window_pos];
            else
                c->output[in_pos - c->impulse_resp_size] = c->out_time_buffer[window_pos] / c->window_size;
        }
    }
}
