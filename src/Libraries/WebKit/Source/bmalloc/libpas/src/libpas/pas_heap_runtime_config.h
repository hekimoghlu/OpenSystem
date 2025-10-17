/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#ifndef PAS_HEAP_RUNTIME_CONFIG_H
#define PAS_HEAP_RUNTIME_CONFIG_H

#include "pas_page_sharing_mode.h"
#include "pas_segregated_heap_lookup_kind.h"
#include "pas_segregated_page_config_variant.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_heap_runtime_config;
struct pas_segregated_page_config;
typedef struct pas_heap_runtime_config pas_heap_runtime_config;
typedef struct pas_segregated_page_config pas_segregated_page_config;

typedef size_t (*pas_heap_runtime_config_view_cache_capacity_for_object_size_callback)(
    size_t object_size, const pas_segregated_page_config* page_config);

struct pas_heap_runtime_config {
    pas_segregated_heap_lookup_kind lookup_kind : 8;
    pas_page_sharing_mode sharing_mode : 8;

    bool statically_allocated : 1;
    bool is_part_of_heap : 1;
    
    unsigned directory_size_bound_for_partial_views;
    unsigned directory_size_bound_for_baseline_allocators;
    unsigned directory_size_bound_for_no_view_cache;

    /* It's OK to set these to UINT_MAX, in which case the maximum is decided by the heap_config. */
    unsigned max_segregated_object_size;
    unsigned max_bitfit_object_size;

    pas_heap_runtime_config_view_cache_capacity_for_object_size_callback view_cache_capacity_for_object_size;
};

PAS_API uint8_t pas_heap_runtime_config_view_cache_capacity_for_object_size(
    pas_heap_runtime_config* config,
    size_t object_size,
    const pas_segregated_page_config* page_config);

PAS_API size_t pas_heap_runtime_config_zero_view_cache_capacity(
    size_t object_size,
    const pas_segregated_page_config* page_config);

PAS_API size_t pas_heap_runtime_config_aggressive_view_cache_capacity(
    size_t object_size,
    const pas_segregated_page_config* page_config);

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_RUNTIME_CONFIG_H */

