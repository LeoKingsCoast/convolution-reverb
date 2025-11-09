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

#define IMPULSE_RESP_SIZE 1024
#define FRAMES_PER_BUFFER 1024
#define SAMPLE_RATE 44100

typedef struct user_data {
    ConvolutorCircular* convolutor;
    float* overlap_buffer;
    int impulse_resp_size;
    RingBuffer* ring_buffer;
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

    ConvolutorCircular* conv = params->convolutor;
    RingBuffer* ring_buffer = params->ring_buffer;
    float* overlap_buffer = params->overlap_buffer;
    int impulse_resp_size = params->impulse_resp_size;

    if (!in || !out)
        return paContinue;

    memcpy(conv->input, overlap_buffer, (impulse_resp_size - 1) * sizeof(float));
    memcpy(conv->input + (impulse_resp_size - 1), in, frames_per_buffer * sizeof(float));

    conv_circular(conv);

    for (int i = 0; i < frames_per_buffer; i++) {
        float sample = conv->output[impulse_resp_size - 1 + i];
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

int main() {
    // ================= LOCK MEMORY ==================

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("Failed to lock memory");
        return -1;
    }


    // =========== CREATE IMPULSE RESPONSE ============

    float impulse_resp[IMPULSE_RESP_SIZE];

    float decay_time = 0.02;    // seconds until ~-60dB (approx)
    float tau = decay_time / logf(1000.0f);

    for (int i = 0; i < IMPULSE_RESP_SIZE; i++) {
        float t = i / (float) SAMPLE_RATE;
        impulse_resp[i] = 0.05 * expf(-t / tau);
    }

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
        .convolutor = conv_init_circular(FRAMES_PER_BUFFER + IMPULSE_RESP_SIZE - 1, IMPULSE_RESP_SIZE, impulse_resp),
        .overlap_buffer = calloc((IMPULSE_RESP_SIZE - 1), sizeof(float)),
        .impulse_resp_size = IMPULSE_RESP_SIZE,
        .ring_buffer = ring_buffer,
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
    conv_terminate_circular(user_data.convolutor);

    Pa_CloseStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "[Error]: Failed to close stream: %s\n", Pa_GetErrorText(err));
        return -1;
    }
    Pa_Terminate();

    return 0;
}
