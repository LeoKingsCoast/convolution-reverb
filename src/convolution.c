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

static void complex_multiply(fftwf_complex result, const fftwf_complex a, const fftwf_complex b) {
        result[0] = a[0] * b[0] - a[1] * b[1];
        result[1] = a[0] * b[1] + a[1] * b[0];
}

Convolutor* conv_init_linear(int input_size, int impulse_resp_size, float* impulse_resp) {
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
    c->output = malloc(c->output_size * sizeof(float));

    c->impulse_freq_padded = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));

    c->in_time_buffer = malloc(c->window_size * sizeof(float));
    c->in_freq_buffer = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->out_time_buffer = malloc(c->window_size * sizeof(float));
    c->out_freq_buffer = fftwf_malloc((c->window_size / 2 + 1) * sizeof(fftwf_complex));
    c->fft_plan = fftwf_plan_dft_r2c_1d(c->window_size, c->in_time_buffer, c->out_freq_buffer, FFTW_MEASURE);
    c->ifft_plan = fftwf_plan_dft_c2r_1d(c->window_size, c->in_freq_buffer, c->out_time_buffer, FFTW_MEASURE);

    if (!c->input || !c->window || !c->overlap || !c->output ||
        !c->impulse_freq_padded || !c->in_time_buffer || !c->in_freq_buffer ||
        !c->out_time_buffer || !c->out_freq_buffer || !c->fft_plan || !c->ifft_plan)
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

ConvolutorCircular* conv_init_circular(int input_size, int impulse_resp_size, float* impulse_resp) {
    if (input_size <= 0 || impulse_resp_size <= 0 || impulse_resp == NULL)
        return NULL;

    ConvolutorCircular* c = (ConvolutorCircular*) malloc(sizeof(ConvolutorCircular));
    if (!c)
        return NULL;

    c->input_size = input_size;

    c->input = malloc(c->input_size * sizeof(float));
    c->output = malloc(c->input_size * sizeof(float));

    c->impulse_freq_padded = fftwf_malloc((c->input_size / 2 + 1) * sizeof(fftwf_complex));

    c->in_time_buffer = malloc(c->input_size * sizeof(float));
    c->in_freq_buffer = fftwf_malloc((c->input_size / 2 + 1) * sizeof(fftwf_complex));
    c->out_time_buffer = malloc(c->input_size * sizeof(float));
    c->out_freq_buffer = fftwf_malloc((c->input_size / 2 + 1) * sizeof(fftwf_complex));
    c->fft_plan = fftwf_plan_dft_r2c_1d(c->input_size, c->in_time_buffer, c->out_freq_buffer, FFTW_MEASURE);
    c->ifft_plan = fftwf_plan_dft_c2r_1d(c->input_size, c->in_freq_buffer, c->out_time_buffer, FFTW_MEASURE);

    if (!c->input || !c->output || !c->impulse_freq_padded ||
        !c->in_time_buffer || !c->in_freq_buffer || !c->out_time_buffer ||
        !c->out_freq_buffer || !c->fft_plan || !c->ifft_plan)
    {
        free(c);
        return NULL;
    }

    // Copy the impulse response to the fft input, padding with 0 until the window size
    for (int i = 0; i < c->input_size; i++) {
        if (i < impulse_resp_size) c->in_time_buffer[i] = impulse_resp[i];
        else c->in_time_buffer[i] = 0;
    }

    // impulse response DFT stored in out_freq_buffer
    fftwf_execute(c->fft_plan);
    memcpy(c->impulse_freq_padded, c->out_freq_buffer, (c->input_size / 2 + 1) * sizeof(fftwf_complex));

    return c;
}

void conv_terminate_linear(Convolutor* c) {
    if (c) {
        fftwf_destroy_plan(c->ifft_plan);
        fftwf_destroy_plan(c->fft_plan);
        fftwf_free(c->out_freq_buffer);
        free(c->out_time_buffer);
        fftwf_free(c->in_freq_buffer);
        free(c->in_time_buffer);
        fftwf_free(c->impulse_freq_padded);
        free(c->input);
        free(c->impulse_resp);
        free(c->window);
        free(c->overlap);
        free(c->output);

        free(c);
    }
}

void conv_terminate_circular(ConvolutorCircular* c) {
    if (c) {
        fftwf_destroy_plan(c->ifft_plan);
        fftwf_destroy_plan(c->fft_plan);
        fftwf_free(c->out_freq_buffer);
        free(c->out_time_buffer);
        fftwf_free(c->in_freq_buffer);
        free(c->in_time_buffer);
        fftwf_free(c->impulse_freq_padded);
        free(c->input);
        free(c->output);

        free(c);
    }
}

void conv_linear(Convolutor* c) {
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

void conv_circular(ConvolutorCircular* c) {
    for (int i = 0; i < c->input_size; i++) {
        c->in_time_buffer[i] = c->input[i];
    }

    // Store DFT of this window in out_freq_buffer
    fftwf_execute(c->fft_plan);

    /*
     * Multiply window frequency signal by the impulse response spectrum
     * Complex Multiplication: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
     */
    for (int i = 0; i < (c->input_size / 2 + 1); i++) {
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

    for (int i = 0; i < c->input_size; i++) {
        c->output[i] = c->out_time_buffer[i] / c->input_size;
    }
}

static FDL* fdl_malloc_(int block_size, int n_partitions) {
    const int conv_size = 2 * block_size;

    FDL* fdl = malloc(sizeof(*fdl));
    if (!fdl)
        return NULL;

    fdl->block_size = block_size;
    fdl->n_partitions = n_partitions;
    fdl->transform_size = block_size * 2;

    fdl->filter = malloc(fdl->n_partitions * sizeof(*fdl->filter));
    if (!fdl->filter)
        goto err_free_fdl;

    fdl->delayed_signals = malloc(fdl->n_partitions * sizeof(*fdl->delayed_signals));
    if (!fdl->delayed_signals)
        goto err_free_filter;

    for (int i = 0; i < fdl->n_partitions; i++) {
        fdl->filter[i] = fftwf_malloc((conv_size / 2 + 1) * sizeof(*fdl->filter[i]));
        fdl->delayed_signals[i] = fftwf_malloc((conv_size / 2 + 1) * sizeof(*fdl->delayed_signals[i]));

        if (!fdl->filter[i] || !fdl->delayed_signals[i]) {
            for (int j = 0; j < i; j++) {
                if (fdl->filter[j]) fftwf_free(fdl->filter[j]);
                if (fdl->delayed_signals[j]) fftwf_free(fdl->delayed_signals[j]);
            }
            goto err_free_delayed_signals;
        }
    }

    fdl->fft_time_buffer = malloc(conv_size * sizeof(*fdl->fft_time_buffer));
    fdl->fft_freq_buffer = fftwf_malloc((conv_size / 2 + 1) * sizeof(*fdl->fft_freq_buffer));
    fdl->ifft_freq_buffer = fftwf_malloc((conv_size / 2 + 1) * sizeof(*fdl->ifft_freq_buffer));
    fdl->ifft_time_buffer = malloc(conv_size * sizeof(*fdl->ifft_time_buffer));

    fdl->out_buffer = malloc(block_size * sizeof(*fdl->out_buffer));

    return fdl;

err_free_delayed_signals:
    free(fdl->delayed_signals);

err_free_filter:
    free(fdl->filter);

err_free_fdl:
    free(fdl);
    return NULL;
}

FDL* conv_init_fdl(int block_size, int n_partitions, float** filter_parts) {
    FDL* fdl = fdl_malloc_(block_size, n_partitions);
    if (!fdl) {
        return NULL;
    }

    const int transform_size = 2 * fdl->block_size;

    fdl->fft_plan = fftwf_plan_dft_r2c_1d(transform_size, fdl->fft_time_buffer, fdl->fft_freq_buffer, FFTW_MEASURE);
    fdl->ifft_plan = fftwf_plan_dft_c2r_1d(transform_size, fdl->ifft_freq_buffer, fdl->ifft_time_buffer, FFTW_MEASURE);

    // Copy filter frequency responses to each partition
    for (int i = 0; i < n_partitions; i++) {
        memcpy(fdl->fft_time_buffer, filter_parts[i], block_size * sizeof(*filter_parts[0]));
        memset(fdl->fft_time_buffer + block_size, 0, block_size * sizeof(*filter_parts[0]));
        fftwf_execute(fdl->fft_plan);
        memcpy(fdl->filter[i], fdl->fft_freq_buffer, (transform_size / 2 + 1) * sizeof(*fdl->fft_freq_buffer));
    }

    memset(fdl->fft_time_buffer, 0, fdl->transform_size * sizeof(fdl->fft_time_buffer[0]));

    return fdl;
}

void conv_terminate_fdl(FDL* fdl) {
    if (fdl) {
        for (int i = 0; i < fdl->n_partitions; i++) {
            fftwf_free(fdl->filter[i]);
            fftwf_free(fdl->delayed_signals[i]);
        }
        free(fdl->filter);
        free(fdl->fft_time_buffer);
        fftwf_free(fdl->fft_freq_buffer);
        free(fdl->ifft_time_buffer);
        fftwf_free(fdl->ifft_freq_buffer);
        free(fdl->out_buffer);
        fftwf_destroy_plan(fdl->fft_plan);
        fftwf_destroy_plan(fdl->ifft_plan);
    }
}

void conv_fdl_process(const float* input, FDL* fdl) {
    /* 
     * Shift previous input before copying the new one. We can't use more 
     * efficient data structures, since we need to pass it to FFTW.
     */
    memmove(fdl->fft_time_buffer, fdl->fft_time_buffer + fdl->block_size, fdl->block_size * sizeof(fdl->fft_time_buffer[0]));
    memcpy(fdl->fft_time_buffer + fdl->block_size, input, fdl->block_size * sizeof(fdl->fft_time_buffer[0]));

    fftwf_execute(fdl->fft_plan);

    for (int i = fdl->n_partitions - 1; i > 0; i--) {
        memcpy(fdl->delayed_signals[i], fdl->delayed_signals[i - 1], (fdl->transform_size / 2 + 1) * sizeof(*fdl->delayed_signals[0]));
    }
    memcpy(fdl->delayed_signals[0], fdl->fft_freq_buffer, (fdl->transform_size / 2 + 1) * sizeof(*fdl->delayed_signals[0]));

    // We use the IFFT input buffer to accumulate the complex multiply results
    for (int i = 0; i < (fdl->transform_size / 2 + 1); i++) {
        fdl->ifft_freq_buffer[i][0] = 0;
        fdl->ifft_freq_buffer[i][1] = 0;
    }

    for (int i = 0; i < fdl->n_partitions; i++) {
        for (int j = 0; j < (fdl->transform_size / 2 + 1); j++) {
            fftwf_complex result;
            complex_multiply(result, fdl->delayed_signals[i][j], fdl->filter[i][j]);
            fdl->ifft_freq_buffer[j][0] += result[0];
            fdl->ifft_freq_buffer[j][1] += result[1];
        }
    }

    fftwf_execute(fdl->ifft_plan);

    // We divide by the transform size here, since the IFFT result is not normalized
    for (int i = 0; i < fdl->block_size; i++) {
        fdl->out_buffer[i] = fdl->ifft_time_buffer[fdl->block_size + i] / fdl->transform_size;
    }
}
