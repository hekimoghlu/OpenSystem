/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef PAS_DEBUG_SPECTRUM_H
#define PAS_DEBUG_SPECTRUM_H

#include "pas_ptr_hash_map.h"

PAS_BEGIN_EXTERN_C;

struct pas_debug_spectrum_entry;
struct pas_stream;
typedef struct pas_debug_spectrum_entry pas_debug_spectrum_entry;
typedef struct pas_stream pas_stream;

typedef void (*pas_debug_spectrum_dump_key)(pas_stream* stream, void* key);

struct pas_debug_spectrum_entry {
    pas_debug_spectrum_dump_key dump;
    uint64_t count;
};

#define PAS_DEBUG_SPECTRUM_USE_FOR_COMMIT 0

PAS_API extern pas_ptr_hash_map pas_debug_spectrum;

PAS_API void pas_debug_spectrum_add(
    void* key, pas_debug_spectrum_dump_key dump, uint64_t count);

PAS_API void pas_debug_spectrum_dump(pas_stream* stream);

/* This resets all counts to zero. However, it doesn't forget about the things that are already in
   the spectrum, so you can't use this to change your mind about dump methods. */
PAS_API void pas_debug_spectrum_reset(void);

PAS_END_EXTERN_C;

#endif /* PAS_DEBUG_SPECTRUM_H */

