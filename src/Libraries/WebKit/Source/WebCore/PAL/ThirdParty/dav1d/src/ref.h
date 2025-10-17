/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef DAV1D_SRC_REF_H
#define DAV1D_SRC_REF_H

#include "dav1d/dav1d.h"

#include "src/mem.h"
#include "src/thread.h"

#include <stdatomic.h>
#include <stddef.h>

struct Dav1dRef {
    void *data;
    const void *const_data;
    atomic_int ref_cnt;
    int free_ref;
    void (*free_callback)(const uint8_t *data, void *user_data);
    void *user_data;
};

Dav1dRef *dav1d_ref_create(size_t size);
Dav1dRef *dav1d_ref_create_using_pool(Dav1dMemPool *pool, size_t size);
Dav1dRef *dav1d_ref_wrap(const uint8_t *ptr,
                         void (*free_callback)(const uint8_t *data, void *user_data),
                         void *user_data);
void dav1d_ref_inc(Dav1dRef *ref);
void dav1d_ref_dec(Dav1dRef **ref);

int dav1d_ref_is_writable(Dav1dRef *ref);

#endif /* DAV1D_SRC_REF_H */
