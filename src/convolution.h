#include <fftw3.h>

typedef struct convolutor {
    double* impulse_resp; // Impuse response time signal
    fftw_complex* impulse_freq_padded; // Impuse response frequency signal, of the time signal padded to the window size
    int impulse_resp_size;

    double* input; // Input time signal
    int input_size; // Input frequency signal

    double* output; // Output time signal
    int output_size; // Output frequency signal

    // --- Auxiliary variables ----
    double* window;
    fftw_complex* window_freq;
    int window_size;

    double* out_block;
    double* out_block_freq;

    double* overlap;
    int overlap_size;

    double* in_time_buffer;
    fftw_complex* in_freq_buffer;
    double* out_time_buffer;
    fftw_complex* out_freq_buffer;

    fftw_plan fft_plan;
    fftw_plan ifft_plan;
} Convolutor;

/**
 * @brief Initializes a Convolutor for a given real input time signal and a 
 * real impulse response time signal. The input should be filled before doing 
 * operations with the Convolutor. Returns null on failure.
 */
Convolutor* conv_init(int input_size, int impulse_resp_size, double* impulse_resp);

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
