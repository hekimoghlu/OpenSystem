/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

#include "src/ref.h"

static void default_free_callback(const uint8_t *const data, void *const user_data) {
    assert(data == user_data);
    dav1d_free_aligned(user_data);
}

Dav1dRef *dav1d_ref_create(size_t size) {
    size = (size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);

    uint8_t *const data = dav1d_alloc_aligned(size + sizeof(Dav1dRef), 64);
    if (!data) return NULL;

    Dav1dRef *const res = (Dav1dRef*)(data + size);
    res->const_data = res->user_data = res->data = data;
    atomic_init(&res->ref_cnt, 1);
    res->free_ref = 0;
    res->free_callback = default_free_callback;

    return res;
}

static void pool_free_callback(const uint8_t *const data, void *const user_data) {
    dav1d_mem_pool_push((Dav1dMemPool*)data, user_data);
}

Dav1dRef *dav1d_ref_create_using_pool(Dav1dMemPool *const pool, size_t size) {
    size = (size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);

    Dav1dMemPoolBuffer *const buf =
        dav1d_mem_pool_pop(pool, size + sizeof(Dav1dRef));
    if (!buf) return NULL;

    Dav1dRef *const res = &((Dav1dRef*)buf)[-1];
    res->data = buf->data;
    res->const_data = pool;
    atomic_init(&res->ref_cnt, 1);
    res->free_ref = 0;
    res->free_callback = pool_free_callback;
    res->user_data = buf;

    return res;
}

Dav1dRef *dav1d_ref_wrap(const uint8_t *const ptr,
                         void (*free_callback)(const uint8_t *data, void *user_data),
                         void *const user_data)
{
    Dav1dRef *res = malloc(sizeof(Dav1dRef));
    if (!res) return NULL;

    res->data = NULL;
    res->const_data = ptr;
    atomic_init(&res->ref_cnt, 1);
    res->free_ref = 1;
    res->free_callback = free_callback;
    res->user_data = user_data;

    return res;
}

void dav1d_ref_inc(Dav1dRef *const ref) {
    atomic_fetch_add(&ref->ref_cnt, 1);
}

void dav1d_ref_dec(Dav1dRef **const pref) {
    assert(pref != NULL);

    Dav1dRef *const ref = *pref;
    if (!ref) return;

    if (atomic_fetch_sub(&ref->ref_cnt, 1) == 1) {
        const int free_ref = ref->free_ref;
        ref->free_callback(ref->const_data, ref->user_data);
        if (free_ref) free(ref);
    }
    *pref = NULL;
}

int dav1d_ref_is_writable(Dav1dRef *const ref) {
    return atomic_load(&ref->ref_cnt) == 1 && ref->data;
}
