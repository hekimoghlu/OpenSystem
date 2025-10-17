/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#ifndef MINALIGN32_HEAP_CONFIG_H
#define MINALIGN32_HEAP_CONFIG_H

#include "pas_config.h"

#if PAS_ENABLE_MINALIGN32

#include "pas_heap_config_utils.h"
#include "pas_megapage_cache.h"
#include "pas_simple_type.h"
#include "pas_segregated_page.h"
#include "pas_segregated_page_config_utils.h"

PAS_BEGIN_EXTERN_C;

#define MINALIGN32_MINALIGN_SHIFT ((size_t)5)
#define MINALIGN32_MINALIGN_SIZE ((size_t)1 << MINALIGN32_MINALIGN_SHIFT)

#define MINALIGN32_SMALL_SHARING_SHIFT 3

PAS_API void minalign32_heap_config_activate(void);

#define MINALIGN32_HEAP_CONFIG PAS_BASIC_HEAP_CONFIG( \
    minalign32, \
    .activate = minalign32_heap_config_activate, \
    .get_type_size = pas_simple_type_as_heap_type_get_type_size, \
    .get_type_alignment = pas_simple_type_as_heap_type_get_type_alignment, \
    .dump_type = pas_simple_type_as_heap_type_dump, \
    .check_deallocation = false, \
    .small_segregated_min_align_shift = MINALIGN32_MINALIGN_SHIFT, \
    .small_segregated_sharing_shift = MINALIGN32_SMALL_SHARING_SHIFT, \
    .small_segregated_page_size = PAS_SMALL_PAGE_DEFAULT_SIZE, \
    .small_segregated_wasteage_handicap = PAS_SMALL_PAGE_HANDICAP, \
    .small_exclusive_segregated_logging_mode = pas_segregated_deallocation_size_oblivious_logging_mode, \
    .small_shared_segregated_logging_mode = pas_segregated_deallocation_size_oblivious_logging_mode, \
    .small_exclusive_segregated_enable_empty_word_eligibility_optimization = true, \
    .small_shared_segregated_enable_empty_word_eligibility_optimization = false, \
    .small_segregated_use_reversed_current_word = true, \
    .enable_view_cache = true, \
    .use_small_bitfit = true, \
    .small_bitfit_min_align_shift = MINALIGN32_MINALIGN_SHIFT, \
    .small_bitfit_page_size = PAS_SMALL_BITFIT_PAGE_DEFAULT_SIZE, \
    .medium_page_size = PAS_MEDIUM_PAGE_DEFAULT_SIZE, \
    .granule_size = PAS_GRANULE_DEFAULT_SIZE, \
    .use_medium_segregated = true, \
    .medium_segregated_min_align_shift = PAS_MIN_MEDIUM_ALIGN_SHIFT, \
    .medium_segregated_sharing_shift = PAS_MEDIUM_SHARING_SHIFT, \
    .medium_segregated_wasteage_handicap = PAS_MEDIUM_PAGE_HANDICAP, \
    .medium_exclusive_segregated_logging_mode = pas_segregated_deallocation_size_aware_logging_mode, \
    .medium_shared_segregated_logging_mode = pas_segregated_deallocation_size_aware_logging_mode, \
    .use_medium_bitfit = true, \
    .medium_bitfit_min_align_shift = PAS_MIN_MEDIUM_ALIGN_SHIFT, \
    .use_marge_bitfit = true, \
    .marge_bitfit_min_align_shift = PAS_MIN_MARGE_ALIGN_SHIFT, \
    .marge_bitfit_page_size = PAS_MARGE_PAGE_DEFAULT_SIZE, \
    .pgm_enabled = false)

PAS_API extern const pas_heap_config minalign32_heap_config;

PAS_BASIC_HEAP_CONFIG_DECLARATIONS(minalign32, MINALIGN32);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_MINALIGN32 */

#endif /* MINALIGN32_HEAP_CONFIG_H */

