#include <fftw3.h>

typedef struct convolutor {
    float* impulse_resp; // Impuse response time signal
    fftwf_complex* impulse_freq_padded; // Impuse response frequency signal, of the time signal padded to the window size
    int impulse_resp_size;

    float* input; // Input time signal
    int input_size; // Input frequency signal

    float* output; // Output time signal
    int output_size; // Output frequency signal

    // --- Auxiliary variables ----
    float* window;
    fftwf_complex* window_freq;
    int window_size;

    float* out_block;
    float* out_block_freq;

    float* overlap;
    int overlap_size;

    float* in_time_buffer;
    fftwf_complex* in_freq_buffer;
    float* out_time_buffer;
    fftwf_complex* out_freq_buffer;

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan;
} Convolutor;

/**
 * @brief Initializes a Convolutor for a given real input time signal and a 
 * real impulse response time signal. The input should be filled before doing 
 * operations with the Convolutor. Returns null on failure.
 */
Convolutor* conv_init(int input_size, int impulse_resp_size, float* impulse_resp);

/**
 * @brief Calculates the convolution between the input and imppulse_resp buffers
 * in the Convolutor object. The result is stored in the output buffer. The
 * input and impulse_resp buffers should be filled before calling this function.
 */
void conv_convolve(Convolutor*);

/**
 * @brief Terminates the convolutor object. Returns 0 on success and -1 on failure.
 */
void conv_terminate(Convolutor*);
