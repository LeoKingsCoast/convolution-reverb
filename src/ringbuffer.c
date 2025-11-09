#include "ringbuffer.h"
#include <stddef.h>
#include <stdio.h>

/*
* Return the next index inside the ring buffer. 'i' should be either the head 
* or tail, and 'max_size' should be the buffer size.
*/
static int next_index(int i, int max_size) {
    int next = (i + 1 < max_size + 1) ? i + 1 : 0;
    return next;
}

RingBuffer* ring_buffer_init(size_t size) {
    RingBuffer* r = malloc(sizeof(RingBuffer));
    if (!r)
        return NULL;

    r->data = malloc((size + 1) * sizeof(float));
    if (!r->data) {
        free(r);
        return NULL;
    }

    r->head = 0, r->tail = 0;
    r->size = size;

    return r;
}

int ring_buffer_push(RingBuffer* r, float new_item) {
    if (!ring_buffer_full(r)) {
        r->data[r->head] = new_item;
        r->head = next_index(r->head, r->size);
        return 0;
    }
    else
        return -1;
}

int ring_buffer_pop(RingBuffer* r, float* item) {
    if (!ring_buffer_empty(r)) {
        *item = r->data[r->tail];
        r->tail = next_index(r->tail, r->size);
        return 0;
    }
    else
        return -1;
}

bool ring_buffer_empty(RingBuffer* r) {
    return r->head == r->tail;
}

bool ring_buffer_full(RingBuffer* r) {
    return next_index(r->head, r->size) == r->tail;
}

void ring_buffer_terminate(RingBuffer* r) {
    if (r->data) {
        free(r->data);
        r->data = NULL;
    }

    if (r)
        free(r);
}

void ring_buffer_print(RingBuffer* r) {
    if (!ring_buffer_empty(r)) {
        int i = r->tail;
        printf("{ %f", r->data[i]);
        for (i = r->tail + 1; i != r->head; i = next_index(i, r->size)) {
            printf(", %f", r->data[i]);
        }
        printf(" }\n");
    }
    else
        printf("{ }\n");
}
