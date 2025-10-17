/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <SDL.h>
#include <assert.h>

#include "dp_fifo.h"

// FIFO structure
struct dp_fifo
{
    SDL_mutex *lock;
    SDL_cond *cond_change;
    size_t capacity;
    size_t count;
    void **entries;
    int push_wait;
    int flush;
};


Dav1dPlayPtrFifo *dp_fifo_create(size_t capacity)
{
    Dav1dPlayPtrFifo *fifo;

    assert(capacity > 0);
    if (capacity <= 0)
        return NULL;

    fifo = malloc(sizeof(*fifo));
    if (fifo == NULL)
        return NULL;

    fifo->capacity = capacity;
    fifo->count = 0;
    fifo->push_wait = 0;
    fifo->flush = 0;

    fifo->lock = SDL_CreateMutex();
    if (fifo->lock == NULL) {
        free(fifo);
        return NULL;
    }
    fifo->cond_change = SDL_CreateCond();
    if (fifo->cond_change == NULL) {
        SDL_DestroyMutex(fifo->lock);
        free(fifo);
        return NULL;
    }

    fifo->entries = calloc(capacity, sizeof(void*));
    if (fifo->entries == NULL) {
        dp_fifo_destroy(fifo);
        return NULL;
    }

    return fifo;
}

// Destroy FIFO
void dp_fifo_destroy(Dav1dPlayPtrFifo *fifo)
{
    assert(fifo->count == 0);
    SDL_DestroyMutex(fifo->lock);
    SDL_DestroyCond(fifo->cond_change);
    free(fifo->entries);
    free(fifo);
}

// Push to FIFO
void dp_fifo_push(Dav1dPlayPtrFifo *fifo, void *element)
{
    SDL_LockMutex(fifo->lock);
    while (fifo->count == fifo->capacity) {
        fifo->push_wait = 1;
        SDL_CondWait(fifo->cond_change, fifo->lock);
        fifo->push_wait = 0;
        if (fifo->flush) {
            SDL_CondSignal(fifo->cond_change);
            SDL_UnlockMutex(fifo->lock);
            return;
        }
    }
    fifo->entries[fifo->count++] = element;
    if (fifo->count == 1)
        SDL_CondSignal(fifo->cond_change);
    SDL_UnlockMutex(fifo->lock);
}

// Helper that shifts the FIFO array
static void *dp_fifo_array_shift(void **arr, size_t len)
{
    void *shifted_element = arr[0];
    for (size_t i = 1; i < len; ++i)
        arr[i-1] = arr[i];
    return shifted_element;
}

// Get item from FIFO
void *dp_fifo_shift(Dav1dPlayPtrFifo *fifo)
{
    SDL_LockMutex(fifo->lock);
    while (fifo->count == 0)
        SDL_CondWait(fifo->cond_change, fifo->lock);
    void *res = dp_fifo_array_shift(fifo->entries, fifo->count--);
    if (fifo->count == fifo->capacity - 1)
        SDL_CondSignal(fifo->cond_change);
    SDL_UnlockMutex(fifo->lock);
    return res;
}

void dp_fifo_flush(Dav1dPlayPtrFifo *fifo, void (*destroy_elem)(void *))
{
    SDL_LockMutex(fifo->lock);
    fifo->flush = 1;
    if (fifo->push_wait) {
        SDL_CondSignal(fifo->cond_change);
        SDL_CondWait(fifo->cond_change, fifo->lock);
    }
    while (fifo->count)
        destroy_elem(fifo->entries[--fifo->count]);
    fifo->flush = 0;
    SDL_UnlockMutex(fifo->lock);
}
