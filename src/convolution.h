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

typedef struct fdl {
    fftwf_complex** filter; // Uniformly partitioned filter frequency response
    int n_partitions;
    int block_size;
    int transform_size;

    float* out_buffer;

    // --- Auxiliary variables ----
    fftwf_complex** delayed_signals;

    float* fft_time_buffer;
    fftwf_complex* fft_freq_buffer;
    float* ifft_time_buffer;
    fftwf_complex* ifft_freq_buffer;

    fftwf_plan fft_plan;
    fftwf_plan ifft_plan;
} FDL;

/**
 * @brief Initializes a Convolutor for a given real input time signal and a 
 * real impulse response time signal. The input should be filled before doing 
 * operations with the Convolutor. Returns null on failure.
 */
Convolutor* conv_init(int input_size, int impulse_resp_size, float* impulse_resp);

ConvolutorCircular* conv_init_circular(int input_size, int impulse_resp_size, float* impulse_resp);

/**
 * @brief Initialize an FDL.
 *
 * @param filter_parts Pointer to an already partitioned filter, divided in 
 *        n_partitions lines of block_size elements
 * @param block_size
 * @param n_partitions
 */
FDL* conv_init_fdl(int block_size, int n_partitions, float** filter_parts);

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
 * @brief Calculate the convolution of the input with the partitioned impulse 
 * response provided in conv_fdl_init, taking into account previous inputs as 
 * well. This is meant to be used with signal streams, avoiding discontinuity 
 * between the result of the current block and the previous one. The result is 
 * stored in fdl->output.
 *
 * @param input Time domain input signal. Must have block_size elements (as 
 *        specified in conv_fdl_init)
 *
 * @param[out] fdl->output Convolution result buffer, with block_size elements
 */
void conv_fdl_process(const float* input, FDL* fdl);

/**
 * @brief Terminates the convolutor object.
 */
void conv_terminate_linear(Convolutor*);

/**
 * @brief Terminates the circular convolutor object.
 */
void conv_terminate_circular(ConvolutorCircular*);

/**
 * @brief Terminate the FDL object.
 */
void conv_terminate_fdl(FDL* fdl);
