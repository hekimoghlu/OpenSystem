/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#ifndef PAS_DESIGNATED_INTRINSIC_HEAP_INLINES_H
#define PAS_DESIGNATED_INTRINSIC_HEAP_INLINES_H

#include "pas_allocator_index.h"
#include "pas_designated_intrinsic_heap.h"
#include "pas_local_allocator.h"
#include "pas_segregated_size_directory_inlines.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_allocator_index
pas_designated_intrinsic_heap_num_allocator_indices(pas_heap_config config)
{
    return PAS_MAX(
        pas_segregated_size_directory_num_allocator_indices_for_config(config.small_segregated_config),
        pas_segregated_size_directory_num_allocator_indices_for_config(config.medium_segregated_config));
}

typedef struct {
    size_t index;
    bool did_succeed;
} pas_designated_index_result;

static PAS_ALWAYS_INLINE pas_designated_index_result
pas_designated_index_result_create_failure(void)
{
    pas_designated_index_result result;
    result.index = 0;
    result.did_succeed = false;
    return result;
}

static PAS_ALWAYS_INLINE pas_designated_index_result
pas_designated_index_result_create_success(size_t index)
{
    pas_designated_index_result result;
    result.index = index;
    result.did_succeed = true;
    return result;
}

static PAS_ALWAYS_INLINE size_t
pas_designated_index_result_get_allocator_index(pas_designated_index_result result,
                                                pas_heap_config config)
{
    PAS_TESTING_ASSERT(result.did_succeed);

    return PAS_LOCAL_ALLOCATOR_UNSELECTED_NUM_INDICES +
        result.index * pas_designated_intrinsic_heap_num_allocator_indices(config);
}

static PAS_ALWAYS_INLINE size_t
pas_designated_intrinsic_heap_num_designated_indices_for_small_config(
    pas_segregated_page_config small_config)
{
    /* FIXME: These constants have to match what we do with set_up_range in
       pas_designated_intrinsic_heap_initialize. */
    switch (pas_segregated_page_config_min_align(small_config)) {
    case 8:
        return 38;
    case 16:
        return 26;
    case 32:
        return 14;
    default:
        PAS_ASSERT(!"Unsupported minalign");
        return 0;
    }
}

static PAS_ALWAYS_INLINE pas_designated_index_result
pas_designated_intrinsic_heap_designated_index_for_small_config(
    size_t index,
    pas_intrinsic_heap_designation_mode designation_mode,
    pas_segregated_page_config small_config)
{
    if (!designation_mode)
        return pas_designated_index_result_create_failure();

    /* NOTE: We could do math here. We choose not to because so far that has proved to be
       a perf problem. */
    if (index <= pas_designated_intrinsic_heap_num_designated_indices_for_small_config(small_config))
        return pas_designated_index_result_create_success(index);

    return pas_designated_index_result_create_failure();
}

static PAS_ALWAYS_INLINE size_t
pas_designated_intrinsic_heap_num_designated_indices(pas_heap_config config)
{
    return pas_designated_intrinsic_heap_num_designated_indices_for_small_config(
        config.small_segregated_config);
}

static PAS_ALWAYS_INLINE pas_designated_index_result pas_designated_intrinsic_heap_designated_index(
    size_t index,
    pas_intrinsic_heap_designation_mode designation_mode,
    pas_heap_config config)
{
    return pas_designated_intrinsic_heap_designated_index_for_small_config(
        index, designation_mode, config.small_segregated_config);
}

PAS_END_EXTERN_C;

#endif /* PAS_DESIGNATED_INTRINSIC_HEAP_INLINES_H */

