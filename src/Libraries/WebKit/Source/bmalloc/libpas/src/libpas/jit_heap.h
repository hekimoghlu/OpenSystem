/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#ifndef JIT_HEAP_H
#define JIT_HEAP_H

#include "pas_config.h"

#if PAS_ENABLE_JIT

#include "pas_allocator_counts.h"
#include "pas_heap_ref.h"
#include "pas_intrinsic_heap_support.h"
#include "pas_range.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_heap jit_common_primitive_heap;
PAS_API extern pas_intrinsic_heap_support jit_common_primitive_heap_support;
PAS_API extern pas_allocator_counts jit_allocator_counts;

/* We expect the given memory to be committed and clean, but it may have weird permissions. */
PAS_API void jit_heap_add_fresh_memory(pas_range range);

PAS_API void* jit_heap_try_allocate(size_t size);
PAS_API void jit_heap_shrink(void* object, size_t new_size);
PAS_API size_t jit_heap_get_size(void* object);
PAS_API void jit_heap_deallocate(void* object);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_JIT */

#endif /* JIT_HEAP_H */

