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
    int window_size;

    float* overlap;
    int overlap_size;

    float* in_time_buffer;
    fftwf_complex* in_freq_buffer;
    float* out_time_buffer;
    fftwf_complex* out_freq_buffer;

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan;
} Convolutor;

typedef struct convolutor_circular {
    fftwf_complex* impulse_freq_padded; // Impuse response frequency signal, of the time signal padded to the window size

    float* input; // Input time signal
    int input_size; // Input frequency signal

    float* output; // Output time signal

    // --- Auxiliary variables ----
    float* in_time_buffer;
    fftwf_complex* in_freq_buffer;
    float* out_time_buffer;
    fftwf_complex* out_freq_buffer;

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan;
} ConvolutorCircular;

/**
 * @brief Initializes a Convolutor for a given real input time signal and a 
 * real impulse response time signal. The input should be filled before doing 
 * operations with the Convolutor. Returns null on failure.
 */
Convolutor* conv_init(int input_size, int impulse_resp_size, float* impulse_resp);

ConvolutorCircular* conv_init_circular(int input_size, int impulse_resp_size, float* impulse_resp);

/**
 * @brief Calculates the convolution of the input and imppulse_resp buffers
 * in the Convolutor object. The result is stored in the output buffer. The
 * input and impulse_resp buffers should be filled before calling this function.
 */
void conv_linear(Convolutor*);

/**
 * @brief Calculates the circular convolution of the input and imppulse_resp 
 * buffers in the Convolutor object. The result is stored in the output buffer. The
 * input and impulse_resp buffers should be filled before calling this function.
 */
void conv_circular(ConvolutorCircular* c);

/**
 * @brief Terminates the convolutor object.
 */
void conv_terminate_linear(Convolutor*);

/**
 * @brief Terminates the circular convolutor object.
 */
void conv_terminate_circular(ConvolutorCircular*);
