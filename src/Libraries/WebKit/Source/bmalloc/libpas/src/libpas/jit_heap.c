/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

#include "jit_heap.h"

#if PAS_ENABLE_JIT

#include "jit_heap_config.h"
#include "pas_deallocate.h"
#include "pas_get_allocation_size.h"
#include "pas_try_allocate_intrinsic.h"
#include "pas_try_shrink.h"

PAS_BEGIN_EXTERN_C;

pas_heap jit_common_primitive_heap = PAS_INTRINSIC_HEAP_INITIALIZER(
    &jit_common_primitive_heap,
    NULL,
    jit_common_primitive_heap_support,
    JIT_HEAP_CONFIG,
    &jit_heap_runtime_config);

pas_intrinsic_heap_support jit_common_primitive_heap_support = PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER;

pas_allocator_counts jit_allocator_counts;

void jit_heap_add_fresh_memory(pas_range range)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_JIT_HEAPS);

    if (verbose)
        pas_log("JIT heap adding memory at %p...%p\n", (void*)range.begin, (void*)range.end);
    
    pas_heap_lock_lock();
    jit_heap_config_add_fresh_memory(range);
    pas_heap_lock_unlock();

    if (verbose)
        pas_log("JIT heap done adding memory\n");
}

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    jit_try_allocate_common_primitive_impl,
    JIT_HEAP_CONFIG,
    &jit_heap_runtime_config,
    &jit_allocator_counts,
    pas_allocation_result_identity,
    &jit_common_primitive_heap,
    &jit_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

void* jit_heap_try_allocate(size_t size)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_JIT_HEAPS);
    void* result;
    if (verbose)
        pas_log("jit heap allocating %zu\n", size);
    result = (void*)jit_try_allocate_common_primitive_impl(size, 1, pas_always_compact_allocation_mode).begin;
    if (verbose)
        pas_log("jit heap done allocating, returning %p\n", result);
    return result;
}

void jit_heap_shrink(void* object, size_t new_size)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_JIT_HEAPS);
    if (verbose)
        pas_log("jit heap trying to shrink %p to %zu\n", object, new_size);
    /* NOTE: the shrink call will fail (return false) for segregated allocations, and that's fine because we
       only use segregated allocations for smaller sizes (so the amount of potential memory savings from
       shrinking is small). */
    pas_try_shrink(object, new_size, JIT_HEAP_CONFIG);
}

size_t jit_heap_get_size(void* object)
{
    return pas_get_allocation_size(object, JIT_HEAP_CONFIG);
}

void jit_heap_deallocate(void* object)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_JIT_HEAPS);
    if (verbose)
        pas_log("jit heap deallocating %p\n", object);
    pas_deallocate(object, JIT_HEAP_CONFIG);
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_JIT */

#endif /* LIBPAS_ENABLED */
