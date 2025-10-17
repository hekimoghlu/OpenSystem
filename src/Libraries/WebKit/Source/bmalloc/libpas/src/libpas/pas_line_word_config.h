/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#ifndef PAS_LINE_WORD_CONFIG_H
#define PAS_LINE_WORD_CONFIG_H

#include "pas_log.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_line_word_config;
typedef struct pas_line_word_config pas_line_word_config;

struct pas_line_word_config {
    uint64_t (*get_line_word)(unsigned* alloc_bits, size_t line_index);
    void (*set_line_word)(unsigned* alloc_bits, size_t line_index, uint64_t value);
    uint64_t (*count_low_zeroes)(uint64_t value);
    uint64_t (*count_high_zeroes)(uint64_t value);
};

#define PAS_LINE_WORD_CONFIG_DEFINE_FUNCTIONS(bits_per_line_word) \
    static PAS_ALWAYS_INLINE uint64_t \
    pas_line_word_config_get_line_word_ ## bits_per_line_word ## _bit( \
        unsigned* alloc_bits, size_t line_index) \
    { \
        return ((uint ## bits_per_line_word ## _t*)alloc_bits)[line_index]; \
    } \
    \
    static PAS_ALWAYS_INLINE void \
    pas_line_word_config_set_line_word_ ## bits_per_line_word ## _bit( \
        unsigned* alloc_bits, size_t line_index, uint64_t value) \
    { \
        static const bool verbose = false; \
        if (verbose) \
            pas_log("Setting %p[%zu] := %llx\n", alloc_bits, line_index, value); \
        ((uint ## bits_per_line_word ## _t*)alloc_bits)[line_index] = \
            (uint ## bits_per_line_word ## _t)value; \
    }

PAS_LINE_WORD_CONFIG_DEFINE_FUNCTIONS(8);
PAS_LINE_WORD_CONFIG_DEFINE_FUNCTIONS(16);
PAS_LINE_WORD_CONFIG_DEFINE_FUNCTIONS(32);
PAS_LINE_WORD_CONFIG_DEFINE_FUNCTIONS(64);

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_low_zeroes_8_bit(uint64_t value)
{
    return __builtin_ctz((unsigned)value);
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_high_zeroes_8_bit(uint64_t value)
{
    unsigned leading_zeroes;
    leading_zeroes = __builtin_clz((unsigned)(value & 0xff));
    PAS_TESTING_ASSERT(leading_zeroes >= 24);
    return leading_zeroes - 24;
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_low_zeroes_16_bit(uint64_t value)
{
    return __builtin_ctz((unsigned)value);
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_high_zeroes_16_bit(uint64_t value)
{
    unsigned leading_zeroes;
    leading_zeroes = __builtin_clz((unsigned)(value & 0xffff));
    PAS_TESTING_ASSERT(leading_zeroes >= 16);
    return leading_zeroes - 16;
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_low_zeroes_32_bit(uint64_t value)
{
    return __builtin_ctz((unsigned)value);
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_high_zeroes_32_bit(uint64_t value)
{
    return __builtin_clz((unsigned)value);
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_low_zeroes_64_bit(uint64_t value)
{
    return __builtin_ctzll(value);
}

static PAS_ALWAYS_INLINE uint64_t
pas_line_word_config_count_high_zeroes_64_bit(uint64_t value)
{
    return __builtin_clzll(value);
}

static PAS_ALWAYS_INLINE void pas_line_word_config_construct(pas_line_word_config* config,
                                                             size_t bits_per_line_word)
{

#define PAS_LINE_WORD_CONFIG_CONSTRUCT_CASE(bits_per_line_word) \
    case bits_per_line_word: \
        config->get_line_word = pas_line_word_config_get_line_word_ ## bits_per_line_word ## _bit; \
        config->set_line_word = pas_line_word_config_set_line_word_ ## bits_per_line_word ## _bit; \
        config->count_low_zeroes = \
            pas_line_word_config_count_low_zeroes_ ## bits_per_line_word ## _bit; \
        config->count_high_zeroes = \
            pas_line_word_config_count_high_zeroes_ ## bits_per_line_word ## _bit; \
        break;

    switch (bits_per_line_word) {
    PAS_LINE_WORD_CONFIG_CONSTRUCT_CASE(8);
    PAS_LINE_WORD_CONFIG_CONSTRUCT_CASE(16);
    PAS_LINE_WORD_CONFIG_CONSTRUCT_CASE(32);
    PAS_LINE_WORD_CONFIG_CONSTRUCT_CASE(64);
    default:
        PAS_ASSERT(!"Bad value for bits_per_line_word");
        break;
    }
}

PAS_END_EXTERN_C;

#endif /* PAS_LINE_WORD_CONFIG_H */

