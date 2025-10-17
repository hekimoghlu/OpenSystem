/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#ifndef PAS_SEGREGATED_DIRECTORY_BIT_REFERENCE_H
#define PAS_SEGREGATED_DIRECTORY_BIT_REFERENCE_H

#include "pas_segregated_directory.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_directory_bit_reference;
typedef struct pas_segregated_directory_bit_reference pas_segregated_directory_bit_reference;

struct pas_segregated_directory_bit_reference {
    pas_segregated_directory_bitvector_segment* segment_ptr;
    size_t index;
    unsigned mask;
    bool is_inline_bit;
};

static inline pas_segregated_directory_bit_reference
pas_segregated_directory_bit_reference_create_null(void)
{
    pas_segregated_directory_bit_reference result;
    result.segment_ptr = NULL;
    result.mask = 0;
    result.is_inline_bit = false;
    result.index = 0;
    return result;
}

static inline pas_segregated_directory_bit_reference
pas_segregated_directory_bit_reference_create_inline(void)
{
    pas_segregated_directory_bit_reference result;
    result.segment_ptr = NULL;
    result.mask = 1;
    result.is_inline_bit = true;
    result.index = 0;
    return result;
}

static inline pas_segregated_directory_bit_reference
pas_segregated_directory_bit_reference_create_out_of_line(
    size_t index,
    pas_segregated_directory_bitvector_segment* segment_ptr,
    unsigned mask)
{
    pas_segregated_directory_bit_reference result;
    result.segment_ptr = segment_ptr;
    result.mask = mask;
    result.is_inline_bit = false;
    result.index = index;
    return result;
}

static inline bool pas_segregated_directory_bit_reference_is_null(
    pas_segregated_directory_bit_reference reference)
{
    return !reference.segment_ptr && !reference.is_inline_bit;
}

static inline bool pas_segregated_directory_bit_reference_is_inline(
    pas_segregated_directory_bit_reference reference)
{
    return reference.is_inline_bit;
}

static inline bool pas_segregated_directory_bit_reference_is_out_of_line(
    pas_segregated_directory_bit_reference reference)
{
    return !!reference.segment_ptr;
}

#define PAS_SEGREGATED_DIRECTORY_BIT_REFERENCE_GET(directory, reference, bit_name) ({ \
        pas_segregated_directory* _directory; \
        pas_segregated_directory_bit_reference _reference; \
        bool _result; \
        \
        _directory = (directory); \
        _reference = (reference); \
        \
        PAS_TESTING_ASSERT(!pas_segregated_directory_bit_reference_is_null(_reference)); \
        \
        if (pas_segregated_directory_bit_reference_is_inline(_reference)) \
            _result = !!(_directory->bits & PAS_SEGREGATED_DIRECTORY_BITS_##bit_name##_MASK); \
        else \
            _result = !!(_reference.segment_ptr->bit_name##_bits & _reference.mask); \
        _result; \
    })

#define PAS_SEGREGATED_DIRECTORY_BIT_REFERENCE_SET(directory, reference, bit_name, value) ({ \
        const bool _verbose = false; \
        \
        pas_segregated_directory* _directory; \
        pas_segregated_directory_bit_reference _reference; \
        bool _value; \
        bool _result; \
        \
        _directory = (directory); \
        _reference = (reference); \
        _value = (value); \
        \
        PAS_TESTING_ASSERT(!pas_segregated_directory_bit_reference_is_null(_reference)); \
        \
        if (pas_segregated_directory_bit_reference_is_inline(_reference)) { \
            for (;;) { \
                unsigned old_bits; \
                unsigned new_bits; \
                \
                old_bits = _directory->bits; \
                if (_value) \
                    new_bits = old_bits | PAS_SEGREGATED_DIRECTORY_BITS_##bit_name##_MASK; \
                else \
                    new_bits = old_bits & ~PAS_SEGREGATED_DIRECTORY_BITS_##bit_name##_MASK; \
                \
                if (old_bits == new_bits) { \
                    _result = false; \
                    break; \
                } \
                \
                if (pas_compare_and_swap_uint32_weak(&_directory->bits, old_bits, new_bits)) { \
                    _result = true; \
                    break; \
                } \
            } \
        } else { \
            for (;;) { \
                unsigned old_bits; \
                unsigned new_bits; \
                \
                old_bits = _reference.segment_ptr->bit_name##_bits; \
                if (_value) \
                    new_bits = old_bits | _reference.mask; \
                else \
                    new_bits = old_bits & ~_reference.mask; \
                \
                if (old_bits == new_bits) { \
                    _result = false; \
                    break; \
                } \
                \
                if (pas_compare_and_swap_uint32_weak( \
                        &_reference.segment_ptr->bit_name##_bits, \
                        old_bits, new_bits)) { \
                    _result = true; \
                    break; \
                } \
            } \
        } \
        \
        if (_verbose && _result) { \
            pas_log("%p[%s, %zu]: setting (ref) to %s at %s:%d %s\n", \
                    _directory, #bit_name, _reference.index, _value ? "true" : "false", \
                    __FILE__, __LINE__, __PRETTY_FUNCTION__); \
        } \
        \
        _result; \
    })

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_DIRECTORY_BIT_REFERENCE_H */

