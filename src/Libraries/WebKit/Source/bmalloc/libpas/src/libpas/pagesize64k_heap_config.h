/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#ifndef PAGESIZE64K_HEAP_CONFIG_H
#define PAGESIZE64K_HEAP_CONFIG_H

#include "pas_config.h"

#if PAS_ENABLE_PAGESIZE64K

#include "pas_heap_config_utils.h"
#include "pas_megapage_cache.h"
#include "pas_simple_type.h"
#include "pas_segregated_page.h"
#include "pas_segregated_page_config_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAGESIZE64K_MINALIGN_SHIFT ((size_t)5)
#define PAGESIZE64K_MINALIGN_SIZE ((size_t)1 << PAGESIZE64K_MINALIGN_SHIFT)

#define PAGESIZE64K_SMALL_PAGE_SIZE 65536
#define PAGESIZE64K_MEDIUM_PAGE_SIZE 262144

PAS_API void pagesize64k_heap_config_activate(void);

#define PAGESIZE64K_HEAP_CONFIG PAS_BASIC_HEAP_CONFIG( \
    pagesize64k, \
    .activate = pagesize64k_heap_config_activate, \
    .get_type_size = pas_simple_type_as_heap_type_get_type_size, \
    .get_type_alignment = pas_simple_type_as_heap_type_get_type_alignment, \
    .dump_type = pas_simple_type_as_heap_type_dump, \
    .check_deallocation = false, \
    .small_segregated_min_align_shift = PAGESIZE64K_MINALIGN_SHIFT, \
    .small_segregated_sharing_shift = PAS_SMALL_SHARING_SHIFT, \
    .small_segregated_page_size = PAGESIZE64K_SMALL_PAGE_SIZE, \
    .small_segregated_wasteage_handicap = PAS_SMALL_PAGE_HANDICAP, \
    .small_exclusive_segregated_logging_mode = pas_segregated_deallocation_size_oblivious_logging_mode, \
    .small_shared_segregated_logging_mode = pas_segregated_deallocation_size_oblivious_logging_mode, \
    .small_exclusive_segregated_enable_empty_word_eligibility_optimization = false, \
    .small_shared_segregated_enable_empty_word_eligibility_optimization = false, \
    .small_segregated_use_reversed_current_word = !PAS_ARM64, \
    .enable_view_cache = false, \
    .use_small_bitfit = true, \
    .small_bitfit_min_align_shift = PAGESIZE64K_MINALIGN_SHIFT, \
    .small_bitfit_page_size = PAGESIZE64K_SMALL_PAGE_SIZE, \
    .medium_page_size = PAGESIZE64K_MEDIUM_PAGE_SIZE, \
    .granule_size = PAS_GRANULE_DEFAULT_SIZE, \
    .use_medium_segregated = true, \
    .medium_segregated_min_align_shift = PAS_MIN_MEDIUM_ALIGN_SHIFT, \
    .medium_segregated_sharing_shift = PAS_MEDIUM_SHARING_SHIFT, \
    .medium_segregated_wasteage_handicap = PAS_MEDIUM_PAGE_HANDICAP, \
    .medium_exclusive_segregated_logging_mode = pas_segregated_deallocation_size_aware_logging_mode, \
    .medium_shared_segregated_logging_mode = pas_segregated_deallocation_size_aware_logging_mode, \
    .use_medium_bitfit = true, \
    .medium_bitfit_min_align_shift = PAS_MIN_MEDIUM_ALIGN_SHIFT, \
    .use_marge_bitfit = false, \
    .pgm_enabled = false)

PAS_API extern const pas_heap_config pagesize64k_heap_config;

PAS_BASIC_HEAP_CONFIG_DECLARATIONS(pagesize64k, PAGESIZE64K);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_PAGESIZE64K */

#endif /* PAGESIZE64K_HEAP_CONFIG_H */

