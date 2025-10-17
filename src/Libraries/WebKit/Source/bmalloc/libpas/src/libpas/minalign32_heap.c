/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "minalign32_heap.h"

#if PAS_ENABLE_MINALIGN32

#include "iso_heap_innards.h"
#include "minalign32_heap_config.h"
#include "pas_deallocate.h"
#include "pas_try_allocate.h"
#include "pas_try_allocate_array.h"
#include "pas_try_allocate_intrinsic.h"

pas_intrinsic_heap_support minalign32_common_primitive_heap_support =
    PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER;

pas_heap minalign32_common_primitive_heap =
    PAS_INTRINSIC_HEAP_INITIALIZER(
        &minalign32_common_primitive_heap,
        PAS_SIMPLE_TYPE_CREATE(1, 1),
        minalign32_common_primitive_heap_support,
        MINALIGN32_HEAP_CONFIG,
        &minalign32_intrinsic_runtime_config.base);

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    test_allocate_common_primitive,
    MINALIGN32_HEAP_CONFIG,
    &minalign32_intrinsic_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error,
    &minalign32_common_primitive_heap,
    &minalign32_common_primitive_heap_support,
    pas_intrinsic_heap_is_designated);

PAS_CREATE_TRY_ALLOCATE(
    test_allocate_impl,
    MINALIGN32_HEAP_CONFIG,
    &minalign32_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error);

PAS_CREATE_TRY_ALLOCATE_ARRAY(
    test_allocate_array_impl,
    MINALIGN32_HEAP_CONFIG,
    &minalign32_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error);

void* minalign32_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)test_allocate_common_primitive(size, 1, allocation_mode).begin;
}

void* minalign32_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    return (void*)test_allocate_impl(heap_ref, allocation_mode).begin;
}

void* minalign32_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)test_allocate_array_impl_by_count(heap_ref, count, alignment, allocation_mode).begin;
}

void minalign32_deallocate(void* ptr)
{
    pas_deallocate(ptr, MINALIGN32_HEAP_CONFIG);
}

pas_heap* minalign32_heap_ref_get_heap(pas_heap_ref* heap_ref)
{
    return pas_ensure_heap(heap_ref, pas_normal_heap_ref_kind,
                           &minalign32_heap_config, &minalign32_typed_runtime_config.base);
}

#endif /* PAS_ENABLE_MINALIGN32 */

#endif /* LIBPAS_ENABLED */
