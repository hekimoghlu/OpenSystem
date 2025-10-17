/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#ifndef PAS_SEGREGATED_PAGE_CONFIG_INLINES_H
#define PAS_SEGREGATED_PAGE_CONFIG_INLINES_H

#include "pas_config.h"
#include "pas_local_allocator_inlines.h"
#include "pas_segregated_page_config.h"

PAS_BEGIN_EXTERN_C;

#define PAS_SEGREGATED_PAGE_CONFIG_TLC_SPECIALIZATION_DEFINITIONS(lower_case_page_config_name, page_config_value) \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_page_config_name ## _specialized_local_allocator_try_allocate_in_primordial_partial_view( \
        pas_local_allocator* allocator, pas_allocation_mode allocation_mode) \
    { \
        return pas_local_allocator_try_allocate_in_primordial_partial_view( \
            allocator, allocation_mode, (page_config_value)); \
    } \
    \
    PAS_NEVER_INLINE bool lower_case_page_config_name ## _specialized_local_allocator_start_allocating_in_primordial_partial_view( \
        pas_local_allocator* allocator, \
        pas_segregated_partial_view* partial, \
        pas_segregated_size_directory* size_directory) \
    { \
        return pas_local_allocator_start_allocating_in_primordial_partial_view( \
            allocator, partial, size_directory, (page_config_value)); \
    } \
    \
    PAS_NEVER_INLINE bool lower_case_page_config_name ## _specialized_local_allocator_refill( \
        pas_local_allocator* allocator, \
        pas_allocator_counts* counts) \
    { \
        return pas_local_allocator_refill_with_known_config(allocator, counts, (page_config_value)); \
    } \
    \
    void lower_case_page_config_name ## _specialized_local_allocator_return_memory_to_page( \
        pas_local_allocator* allocator, \
        pas_segregated_view view, \
        pas_segregated_page* page, \
        pas_segregated_size_directory* directory, \
        pas_lock_hold_mode heap_lock_hold_mode) \
    { \
        pas_local_allocator_return_memory_to_page( \
            allocator, view, page, directory, heap_lock_hold_mode, (page_config_value)); \
    } \
    \
    struct pas_dummy

#define PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATION_DEFINITIONS(lower_case_page_config_name, page_config_value) \
    PAS_SEGREGATED_PAGE_CONFIG_TLC_SPECIALIZATION_DEFINITIONS(lower_case_page_config_name, page_config_value)

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATD_PAGE_CONFIG_INLINES_H */

