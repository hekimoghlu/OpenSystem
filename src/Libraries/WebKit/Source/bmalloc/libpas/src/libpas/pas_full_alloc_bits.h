/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#ifndef PAS_FULL_ALLOC_BITS_H
#define PAS_FULL_ALLOC_BITS_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_full_alloc_bits;
typedef struct pas_full_alloc_bits pas_full_alloc_bits;

struct pas_full_alloc_bits {
    unsigned* bits;
    unsigned word_index_begin;
    unsigned word_index_end;
};

static inline pas_full_alloc_bits pas_full_alloc_bits_create_empty(void)
{
    pas_full_alloc_bits result;
    result.bits = NULL;
    result.word_index_begin = 0;
    result.word_index_end = 0;
    return result;
}

static inline pas_full_alloc_bits pas_full_alloc_bits_create(unsigned* bits,
                                                             unsigned word_index_begin,
                                                             unsigned word_index_end)
{
    pas_full_alloc_bits result;
    result.bits = bits;
    result.word_index_begin = word_index_begin;
    result.word_index_end = word_index_end;
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_FULL_ALLOC_BITS_H */

