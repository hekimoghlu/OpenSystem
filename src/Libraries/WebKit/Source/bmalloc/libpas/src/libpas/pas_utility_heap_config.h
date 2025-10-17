/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#ifndef PAS_UTILITY_HEAP_CONFIG_H
#define PAS_UTILITY_HEAP_CONFIG_H

#include "pas_heap_config_utils.h"
#include "pas_segregated_page.h"
#include "pas_segregated_page_config_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_UTILITY_NUM_ALLOC_BITS \
    PAS_BASIC_SEGREGATED_NUM_ALLOC_BITS(PAS_INTERNAL_MIN_ALIGN_SHIFT, \
                                        PAS_SMALL_PAGE_DEFAULT_SIZE)

#define PAS_UTILITY_HEAP_PAYLOAD_OFFSET \
    PAS_BASIC_SEGREGATED_PAYLOAD_OFFSET_EXCLUSIVE(PAS_INTERNAL_MIN_ALIGN_SHIFT, \
                                                  PAS_SMALL_PAGE_DEFAULT_SIZE, \
                                                  PAS_SMALL_PAGE_DEFAULT_SIZE)

static inline pas_page_base* pas_utility_heap_page_header_for_boundary(void* allocation)
{
    return (pas_page_base*)allocation;
}

static inline void* pas_utility_heap_boundary_for_page_header(pas_page_base* page)
{
    return page;
}

PAS_API void* pas_utility_heap_allocate_page(
    pas_segregated_heap* heap, pas_physical_memory_transaction* transaction, pas_segregated_page_role role);

PAS_API pas_segregated_shared_page_directory*
pas_utility_heap_shared_page_directory_selector(pas_segregated_heap* heap,
                                                pas_segregated_size_directory* directory);

static inline pas_page_base* pas_utility_heap_create_page_header(
    void* boundary, pas_page_kind kind, pas_lock_hold_mode heap_lock_hold_mode)
{
    PAS_UNUSED_PARAM(heap_lock_hold_mode);
    PAS_ASSERT(kind == pas_small_exclusive_segregated_page_kind);
    return (pas_page_base*)boundary;
}

static inline void pas_utility_heap_destroy_page_header(
    pas_page_base* page_base, pas_lock_hold_mode heap_lock_hold_mode)
{
    PAS_UNUSED_PARAM(page_base);
    PAS_UNUSED_PARAM(heap_lock_hold_mode);
}

PAS_API bool pas_utility_heap_config_for_each_shared_page_directory(
    pas_segregated_heap* heap,
    bool (*callback)(pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg);

PAS_API void pas_utility_heap_config_dump_shared_page_directory_arg(
    pas_stream* stream, pas_segregated_shared_page_directory* directory);

#define PAS_UTILITY_HEAP_CONFIG ((pas_heap_config){ \
        .config_ptr = &pas_utility_heap_config, \
        .kind = pas_heap_config_kind_pas_utility, \
        .activate_callback = NULL, \
        .get_type_size = NULL, \
        .get_type_alignment = NULL, \
        .dump_type = NULL, \
        .large_alignment = PAS_INTERNAL_MIN_ALIGN, \
        .small_segregated_config = { \
            .base = { \
                .is_enabled = true, \
                .heap_config_ptr = &pas_utility_heap_config, \
                .page_config_ptr = &pas_utility_heap_config.small_segregated_config.base, \
                .page_config_kind = pas_page_config_kind_segregated, \
                .page_config_size_category = pas_page_config_size_category_small, \
                .min_align_shift = PAS_INTERNAL_MIN_ALIGN_SHIFT, \
                .page_size = PAS_SMALL_PAGE_DEFAULT_SIZE, \
                .granule_size = PAS_SMALL_PAGE_DEFAULT_SIZE, \
                .max_object_size = PAS_UTILITY_LOOKUP_SIZE_UPPER_BOUND, \
                .page_header_for_boundary = pas_utility_heap_page_header_for_boundary, \
                .boundary_for_page_header = pas_utility_heap_boundary_for_page_header, \
                .page_header_for_boundary_remote = NULL, \
                .create_page_header = pas_utility_heap_create_page_header, \
                .destroy_page_header = pas_utility_heap_destroy_page_header, \
            }, \
            .variant = pas_small_segregated_page_config_variant, \
            .kind = pas_segregated_page_config_kind_pas_utility_small, \
            .wasteage_handicap = 1., \
            .sharing_shift = PAS_SMALL_SHARING_SHIFT, \
            .num_alloc_bits = PAS_UTILITY_NUM_ALLOC_BITS, \
            .shared_payload_offset = 0, \
            .exclusive_payload_offset = PAS_UTILITY_HEAP_PAYLOAD_OFFSET, \
            .shared_payload_size = 0, \
            .exclusive_payload_size = \
                PAS_SMALL_PAGE_DEFAULT_SIZE - PAS_UTILITY_HEAP_PAYLOAD_OFFSET, \
            .shared_logging_mode = pas_segregated_deallocation_no_logging_mode, \
            .exclusive_logging_mode = pas_segregated_deallocation_no_logging_mode, \
            .use_reversed_current_word = PAS_ARM64, \
            .check_deallocation = false, \
            .enable_empty_word_eligibility_optimization_for_shared = false, \
            .enable_empty_word_eligibility_optimization_for_exclusive = false, \
            .enable_view_cache = false, \
            .page_allocator = pas_utility_heap_allocate_page, \
            .shared_page_directory_selector = pas_utility_heap_shared_page_directory_selector, \
            PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATIONS(pas_utility_heap_page_config) \
        }, \
        .medium_segregated_config = { \
            .base = { \
                .is_enabled = false \
            } \
        }, \
        .small_bitfit_config = { \
            .base = { \
                .is_enabled = false \
            } \
        }, \
        .medium_bitfit_config = { \
            .base = { \
                .is_enabled = false \
            } \
        }, \
        .marge_bitfit_config = { \
            .base = { \
                .is_enabled = false \
            } \
        }, \
        .small_lookup_size_upper_bound = PAS_UTILITY_LOOKUP_SIZE_UPPER_BOUND, \
        .fast_megapage_kind_func = NULL, \
        .small_segregated_is_in_megapage = false, \
        .small_bitfit_is_in_megapage = false, \
        .page_header_func = NULL, \
        .aligned_allocator = NULL, \
        .aligned_allocator_talks_to_sharing_pool = false, \
        .deallocator = NULL, \
        .mmap_capability = pas_may_mmap, \
        .root_data = NULL, \
        .prepare_to_enumerate = NULL, \
        .for_each_shared_page_directory = pas_utility_heap_config_for_each_shared_page_directory, \
        .for_each_shared_page_directory_remote = NULL, \
        .dump_shared_page_directory_arg = pas_utility_heap_config_dump_shared_page_directory_arg, \
        PAS_HEAP_CONFIG_SPECIALIZATIONS(pas_utility_heap_config) \
    })

PAS_API extern const pas_heap_config pas_utility_heap_config;

PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATION_DECLARATIONS(pas_utility_heap_page_config);
PAS_HEAP_CONFIG_SPECIALIZATION_DECLARATIONS(pas_utility_heap_config);

static PAS_ALWAYS_INLINE bool pas_heap_config_is_utility(const pas_heap_config* config)
{
    return config == &pas_utility_heap_config;
}

static PAS_ALWAYS_INLINE pas_lock_hold_mode pas_heap_config_heap_lock_hold_mode(const pas_heap_config* config)
{
    return pas_heap_config_is_utility(config)
        ? pas_lock_is_held
        : pas_lock_is_not_held;
}

PAS_END_EXTERN_C;

#endif /* PAS_UTILITY_HEAP_CONFIG_H */

