/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#ifndef PAS_HEAP_CONFIG_INLINES_H
#define PAS_HEAP_CONFIG_INLINES_H

#include "pas_deallocate.h"
#include "pas_heap_config.h"
#include "pas_local_allocator_inlines.h"
#include "pas_try_allocate_common.h"

PAS_BEGIN_EXTERN_C;

#define PAS_HEAP_CONFIG_SPECIALIZATION_DEFINITIONS(lower_case_heap_config_name, heap_config_value) \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_heap_config_name ## _specialized_local_allocator_try_allocate_small_segregated_slow( \
        pas_local_allocator* allocator, pas_allocation_mode allocation_mode, pas_allocator_counts* count, \
        pas_allocation_result_filter result_filter) \
    { \
        return pas_local_allocator_try_allocate_small_segregated_slow( \
            allocator, allocation_mode, (heap_config_value), count, result_filter); \
    } \
    \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_heap_config_name ## _specialized_local_allocator_try_allocate_medium_segregated_with_free_bits( \
        pas_local_allocator* allocator, pas_allocation_mode allocation_mode) \
    { \
        return pas_local_allocator_try_allocate_with_free_bits( \
            allocator, allocation_mode, (heap_config_value).medium_segregated_config); \
    } \
    \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_heap_config_name ## _specialized_local_allocator_try_allocate_inline_cases( \
        pas_local_allocator* allocator, pas_allocation_mode allocation_mode) \
    { \
        return pas_local_allocator_try_allocate_inline_cases(allocator, allocation_mode, (heap_config_value)); \
    } \
    \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_heap_config_name ## _specialized_local_allocator_try_allocate_slow( \
        pas_local_allocator* allocator, \
        size_t size, \
        size_t alignment, \
        pas_allocation_mode allocation_mode, \
        pas_allocator_counts* counts, \
        pas_allocation_result_filter result_filter) \
    { \
        return pas_local_allocator_try_allocate_slow( \
            allocator, size, alignment, allocation_mode, (heap_config_value), counts, result_filter); \
    } \
    \
    PAS_NEVER_INLINE pas_allocation_result \
    lower_case_heap_config_name ## _specialized_try_allocate_common_impl_slow( \
        pas_heap_ref* heap_ref, \
        pas_heap_ref_kind heap_ref_kind, \
        size_t size, \
        size_t alignment, \
        pas_allocation_mode allocation_mode, \
        pas_heap_runtime_config* runtime_config, \
        pas_allocator_counts* allocator_counts, \
        pas_size_lookup_mode size_lookup_mode) \
    { \
        return pas_try_allocate_common_impl_slow( \
            heap_ref, heap_ref_kind, size, alignment, allocation_mode, (heap_config_value), \
            runtime_config, allocator_counts, size_lookup_mode); \
    } \
    \
    bool lower_case_heap_config_name ## _specialized_try_deallocate_not_small_exclusive_segregated( \
        pas_thread_local_cache* thread_local_cache, \
        uintptr_t begin, \
        pas_deallocation_mode deallocation_mode, \
        pas_fast_megapage_kind megapage_kind) \
    { \
        return pas_try_deallocate_not_small_exclusive_segregated( \
            thread_local_cache, begin, (heap_config_value), deallocation_mode, megapage_kind); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_CONFIG_INLINES_H */

