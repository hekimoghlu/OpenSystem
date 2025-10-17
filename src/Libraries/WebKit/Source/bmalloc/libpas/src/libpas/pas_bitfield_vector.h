/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#ifndef PAS_BITFIELD_VECTOR_H
#define PAS_BITFIELD_VECTOR_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BITFIELD_VECTOR_BITS_PER_WORD 32

/* A bitfield vector can be used to store bitfields that have a power-of-2 number of bits
   ranging from 1 bit to 32 bits. */

#define PAS_BITFIELD_VECTOR_NUM_WORDS(num_fields, num_bits_per_field) \
    (((num_fields) * (num_bits_per_field) + 31) >> 5)
#define PAS_BITFIELD_VECTOR_NUM_BYTES(num_fields, num_bits_per_field) \
    (PAS_BITFIELD_VECTOR_NUM_WORDS((num_fields), (num_bits_per_field)) * sizeof(unsigned))
#define PAS_BITFIELD_VECTOR_NUM_FIELDS(num_words, num_bits_per_field) \
    (((num_words) << 5) / (num_bits_per_field))

#define PAS_BITFIELD_VECTOR_WORD_INDEX(field_index, num_bits_per_field) \
    (((field_index) * (num_bits_per_field)) >> 5)
#define PAS_BITFIELD_VECTOR_FIELD_INDEX(word_index, num_bits_per_field) \
    (((word_index) << 5) / (num_bits_per_field))
#define PAS_BITFIELD_VECTOR_FIELD_SHIFT(field_index, num_bits_per_field) \
    (((field_index) * (num_bits_per_field)) & 31)
#define PAS_BITFIELD_VECTOR_FIELD_MASK(num_bits_per_field) \
    ((unsigned)((((uint64_t)1) << (num_bits_per_field)) - 1))

static inline unsigned pas_bitfield_vector_get(const unsigned* bits,
                                               unsigned num_bits_per_field,
                                               size_t index)
{
    return (bits[PAS_BITFIELD_VECTOR_WORD_INDEX(index, num_bits_per_field)] >>
            PAS_BITFIELD_VECTOR_FIELD_SHIFT(index, num_bits_per_field)) &
        PAS_BITFIELD_VECTOR_FIELD_MASK(num_bits_per_field);
}

static inline void pas_bitfield_vector_set(unsigned* bits, unsigned num_bits_per_field,
                                           size_t index, unsigned value)
{
    unsigned* ptr;
    unsigned word;
    unsigned mask;
    unsigned shift;
    
    ptr = bits + PAS_BITFIELD_VECTOR_WORD_INDEX(index, num_bits_per_field);
    
    mask = PAS_BITFIELD_VECTOR_FIELD_MASK(num_bits_per_field);
    shift = PAS_BITFIELD_VECTOR_FIELD_SHIFT(index, num_bits_per_field);
    
    PAS_ASSERT(value <= mask);
    
    word = *ptr;

    word &= ~(mask << shift);
    word |= value << shift;
    
    *ptr = word;
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIELD_VECTOR_H */

