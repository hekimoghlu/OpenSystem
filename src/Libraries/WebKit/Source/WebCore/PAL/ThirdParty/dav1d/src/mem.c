/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "config.h"

#include <stdint.h>

#include "src/internal.h"

static COLD void mem_pool_destroy(Dav1dMemPool *const pool) {
    pthread_mutex_destroy(&pool->lock);
    free(pool);
}

void dav1d_mem_pool_push(Dav1dMemPool *const pool, Dav1dMemPoolBuffer *const buf) {
    pthread_mutex_lock(&pool->lock);
    const int ref_cnt = --pool->ref_cnt;
    if (!pool->end) {
        buf->next = pool->buf;
        pool->buf = buf;
        pthread_mutex_unlock(&pool->lock);
        assert(ref_cnt > 0);
    } else {
        pthread_mutex_unlock(&pool->lock);
        dav1d_free_aligned(buf->data);
        if (!ref_cnt) mem_pool_destroy(pool);
    }
}

Dav1dMemPoolBuffer *dav1d_mem_pool_pop(Dav1dMemPool *const pool, const size_t size) {
    assert(!(size & (sizeof(void*) - 1)));
    pthread_mutex_lock(&pool->lock);
    Dav1dMemPoolBuffer *buf = pool->buf;
    pool->ref_cnt++;
    uint8_t *data;
    if (buf) {
        pool->buf = buf->next;
        pthread_mutex_unlock(&pool->lock);
        data = buf->data;
        if ((uintptr_t)buf - (uintptr_t)data != size) {
            /* Reallocate if the size has changed */
            dav1d_free_aligned(data);
            goto alloc;
        }
    } else {
        pthread_mutex_unlock(&pool->lock);
alloc:
        data = dav1d_alloc_aligned(size + sizeof(Dav1dMemPoolBuffer), 64);
        if (!data) {
            pthread_mutex_lock(&pool->lock);
            const int ref_cnt = --pool->ref_cnt;
            pthread_mutex_unlock(&pool->lock);
            if (!ref_cnt) mem_pool_destroy(pool);
            return NULL;
        }
        buf = (Dav1dMemPoolBuffer*)(data + size);
        buf->data = data;
    }

    return buf;
}

COLD int dav1d_mem_pool_init(Dav1dMemPool **const ppool) {
    Dav1dMemPool *const pool = malloc(sizeof(Dav1dMemPool));
    if (pool) {
        if (!pthread_mutex_init(&pool->lock, NULL)) {
            pool->buf = NULL;
            pool->ref_cnt = 1;
            pool->end = 0;
            *ppool = pool;
            return 0;
        }
        free(pool);
    }
    *ppool = NULL;
    return DAV1D_ERR(ENOMEM);
}

COLD void dav1d_mem_pool_end(Dav1dMemPool *const pool) {
    if (pool) {
        pthread_mutex_lock(&pool->lock);
        Dav1dMemPoolBuffer *buf = pool->buf;
        const int ref_cnt = --pool->ref_cnt;
        pool->buf = NULL;
        pool->end = 1;
        pthread_mutex_unlock(&pool->lock);

        while (buf) {
            void *const data = buf->data;
            buf = buf->next;
            dav1d_free_aligned(data);
        }
        if (!ref_cnt) mem_pool_destroy(pool);
    }
}
