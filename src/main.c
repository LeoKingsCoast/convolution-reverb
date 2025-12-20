#include "convolution.h"
#include "audio.h"
#include "ringbuffer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include <fftw3.h>
#include <portaudio.h>

#include <sndfile.h>

#include <sys/mman.h>
#include <pa_linux_alsa.h>

#define IMPULSE_RESP_SIZE 8192
#define BLOCK_SIZE 512
#define FRAMES_PER_BUFFER BLOCK_SIZE
#define SAMPLE_RATE 44100

// Compute the ceiling of x/y. Positive integers only.
#define divide_ceil(x, y) (x % y == 0 ? (x / y) : (1 + x / y))

typedef struct user_data {
    float* overlap_buffer;
    int impulse_resp_size;
    RingBuffer* ring_buffer;
    FDL* fdl;
} UserData;

static int callback(
    const void *in_buffer, void *out_buffer, unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info, PaStreamCallbackFlags status_flags,
    void *user_data
)
{
    const float* in = (const float*) in_buffer;
    float* out = (float*) out_buffer;
    UserData* params = (UserData*) user_data;

    FDL* fdl = params->fdl;
    RingBuffer* ring_buffer = params->ring_buffer;
    float* overlap_buffer = params->overlap_buffer;
    int impulse_resp_size = params->impulse_resp_size;

    if (!in || !out)
        return paContinue;

    conv_fdl_process(in, fdl);

    for (int i = 0; i < frames_per_buffer; i++) {
        float sample = fdl->out_buffer[i];
        ring_buffer_push(ring_buffer, sample);
        *out++ = sample; // left side
        *out++ = sample; // right side
    }

    for (int i = 0; i < impulse_resp_size - 1; i++) {
        overlap_buffer[i] = in[frames_per_buffer - impulse_resp_size + i];
    }

    return paContinue;
}

typedef struct {
    bool* stop_flag;
    SNDFILE* outfile;
    RingBuffer* buffer;
} WritingThreadArgs;

void* wav_write_routine(void* in_args) {
    WritingThreadArgs* args = (WritingThreadArgs*) in_args;
    bool* stop = args->stop_flag;
    SNDFILE* outfile = args->outfile;
    RingBuffer* buffer = args->buffer;

    float sample = 0.0f;
    while (*stop == false) {
        if (ring_buffer_pop(buffer, &sample) < 0) // Ring buffer empty
            continue;
        if (sf_write_float (outfile, &sample, 1) != 1)
            puts (sf_strerror (outfile));
    }

    free(buffer);

    printf("Stop request detected\n");

    return NULL;
}

void fill_impulse_response_reverb(float *ir, int ir_len, float decay_ms, float sample_rate) {
    // To decay by -60dB (10^-3), we need tau = decay_time / ln(10^-3) => tau ~ decay_time / 6.91
    float tau = decay_ms / 1000.0f / 6.91f;
    float sum = 0.0f;
    for (int i = 0; i < ir_len; i++) {
        float t = i / sample_rate;
        ir[i] = expf(-t / tau);
        sum += fabsf(ir[i]);
    }

    // Normalize impulse response to not increase output amplitude on convolution
    if (sum > 0.0f) {
        for (int i = 0; i < ir_len; i++)
            ir[i] /= sum;
    }
}

/**
 * @brief Partition an input signal into a uniformly-partitioned output. Takes
 * in the input signal and a block size. Divides the signal in `n_partitions`
 * partitions of size `block_size`, where `n_partitions` is the next integer
 * >= `in_size / block_size`. The output signal is padded with 0 if the partitions
 * do not fit with the block size
 *
 * @param in Input signal
 * @param in_size Number of elements in the input signal
 * @param block_size Desired block size for each partition of the output
 * @param[out] n_partitions Number of partitions generated
 * @return Uniformly-partitioned output signal
 */
float** get_partitioned_signal(float* in, int in_size, int block_size, int *n_partitions) {
    *n_partitions = divide_ceil(in_size, block_size);
    float** out = malloc(*n_partitions * sizeof(*out));
    if (!out)
        return NULL;
    for (int i = 0; i < *n_partitions; i++) {
        out[i] = malloc(block_size * sizeof(*out[i]));
        if (!out[i]){
            for (int j = 0; j < i; j++)
                if (out[j]) free(out[j]);
            free(out);
            return NULL;
        }
    }

    for (int i = 0; i < in_size; i++) {
        out[i / block_size][i % block_size] = i < in_size
            ? in[i]
            : 0;
    }

    return out;
}

void free_partitioned_signal(float** sig, int n_partitions) {
    if (sig){
        for (int i = 0; i < n_partitions; i++)
            free(sig[i]);
        free(sig);
    }
}

int main() {
    // ================= LOCK MEMORY ==================

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("Failed to lock memory with mlockall");
        return -1;
    }


    // =========== CREATE IMPULSE RESPONSE ============

    float impulse_resp[IMPULSE_RESP_SIZE];

    fill_impulse_response_reverb(impulse_resp, IMPULSE_RESP_SIZE, 160.0, SAMPLE_RATE);

    int n_partitions; // Number of impulse response partitions
    float **partitioned_impulse_resp = get_partitioned_signal(impulse_resp, IMPULSE_RESP_SIZE, BLOCK_SIZE, &n_partitions);

    // =========== CREATE FILE WRITE THREAD ===========

    SF_INFO sfinfo = {};
    sfinfo.samplerate = SAMPLE_RATE;
    sfinfo.channels = 1;
    sfinfo.format = (SF_FORMAT_WAV | SF_FORMAT_FLOAT);

    SNDFILE* outfile = sf_open("out.wav", SFM_WRITE, &sfinfo);
    if (!outfile) {
        fprintf(stderr, "[Error]: Unable to open WAV file for writing.\n");
        return -1;
    }

    RingBuffer* ring_buffer = ring_buffer_init(2048);

    bool writing_stop_flag = false;
    WritingThreadArgs writing_thread_args = {
        .stop_flag = &writing_stop_flag,
        .outfile = outfile,
        .buffer = ring_buffer,
    };

    pthread_t writing_thread;
    if(pthread_create(&writing_thread, NULL, wav_write_routine, &writing_thread_args)) {
        perror("Could not create thread for writing WAV file");
        sf_close(outfile);
        return -1;
    }

    // ================ START STREAM ==================

    UserData user_data = {
        .overlap_buffer = calloc((IMPULSE_RESP_SIZE - 1), sizeof(float)),
        .impulse_resp_size = IMPULSE_RESP_SIZE,
        .ring_buffer = ring_buffer,
        .fdl = conv_init_fdl(BLOCK_SIZE, n_partitions, partitioned_impulse_resp),
    };

    PaStream* stream = portaudio_init( SAMPLE_RATE, paFloat32, FRAMES_PER_BUFFER, callback, &user_data);
    if (!stream) {
        return -1;
    }

    PaAlsa_EnableRealtimeScheduling(stream, true);

    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to start stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }

    printf("Press Enter to stop the stream\n");
    getchar();

    writing_stop_flag = true;
    pthread_join(writing_thread, NULL);

    printf("Closing file...\n");
    sf_close(outfile);

    free(user_data.overlap_buffer);
    conv_terminate_fdl(user_data.fdl);

    Pa_CloseStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to close stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }
    Pa_Terminate();

    free_partitioned_signal(partitioned_impulse_resp, n_partitions);

    return 0;
}
