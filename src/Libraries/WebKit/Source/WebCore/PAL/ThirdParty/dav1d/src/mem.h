/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#ifndef DAV1D_SRC_MEM_H
#define DAV1D_SRC_MEM_H

#include <stdlib.h>

#if defined(HAVE_ALIGNED_MALLOC) || defined(HAVE_MEMALIGN)
#include <malloc.h>
#endif

#include "common/attributes.h"

#include "src/thread.h"

typedef struct Dav1dMemPoolBuffer {
    void *data;
    struct Dav1dMemPoolBuffer *next;
} Dav1dMemPoolBuffer;

typedef struct Dav1dMemPool {
    pthread_mutex_t lock;
    Dav1dMemPoolBuffer *buf;
    int ref_cnt;
    int end;
} Dav1dMemPool;

void dav1d_mem_pool_push(Dav1dMemPool *pool, Dav1dMemPoolBuffer *buf);
Dav1dMemPoolBuffer *dav1d_mem_pool_pop(Dav1dMemPool *pool, size_t size);
int dav1d_mem_pool_init(Dav1dMemPool **pool);
void dav1d_mem_pool_end(Dav1dMemPool *pool);

/*
 * Allocate align-byte aligned memory. The return value can be released
 * by calling the dav1d_free_aligned() function.
 */
static inline void *dav1d_alloc_aligned(size_t sz, size_t align) {
    assert(!(align & (align - 1)));
#ifdef HAVE_POSIX_MEMALIGN
    void *ptr;
    if (posix_memalign(&ptr, align, sz)) return NULL;
    return ptr;
#elif defined(HAVE_ALIGNED_MALLOC)
    return _aligned_malloc(sz, align);
#elif defined(HAVE_MEMALIGN)
    return memalign(align, sz);
#else
#error Missing aligned alloc implementation
#endif
}

static inline void dav1d_free_aligned(void* ptr) {
#ifdef HAVE_POSIX_MEMALIGN
    free(ptr);
#elif defined(HAVE_ALIGNED_MALLOC)
    _aligned_free(ptr);
#elif defined(HAVE_MEMALIGN)
    free(ptr);
#endif
}

static inline void dav1d_freep_aligned(void* ptr) {
    void **mem = (void **) ptr;
    if (*mem) {
        dav1d_free_aligned(*mem);
        *mem = NULL;
    }
}

static inline void freep(void *ptr) {
    void **mem = (void **) ptr;
    if (*mem) {
        free(*mem);
        *mem = NULL;
    }
}

#endif /* DAV1D_SRC_MEM_H */
