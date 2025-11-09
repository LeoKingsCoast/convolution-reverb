#include <stdlib.h>
#include <stdbool.h>

typedef struct ring_buffer {
    float* data;
    int head;
    int tail;
    size_t size;
} RingBuffer;

RingBuffer* ring_buffer_init(size_t);

/**
 * @brief Push an item into the ring buffer, moving its head forward
 *
 * @return 0 on success, -1 on failure (buffer full)
 */
int ring_buffer_push(RingBuffer*, float);

/**
 * @brief Pop an item from the ring buffer, moving its tail forward
 *
 * @param[out] item Output value is written to this address
 *
 * @return 0 on success, -1 on failure (buffer empty)
 */
int ring_buffer_pop(RingBuffer*, float* item);

bool ring_buffer_full(RingBuffer*);

bool ring_buffer_empty(RingBuffer*);

void ring_buffer_print(RingBuffer*);

void ring_buffer_terminate(RingBuffer*);
