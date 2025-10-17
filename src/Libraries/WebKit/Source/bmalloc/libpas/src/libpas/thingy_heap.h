/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#ifndef THINGY_HEAP_H
#define THINGY_HEAP_H

#include "pas_config.h"

#if PAS_ENABLE_THINGY

#include "pas_allocator_counts.h"
#include "pas_heap_ref.h"
#include "pas_intrinsic_heap_support.h"
#include "pas_allocation_mode.h"

#include "thingy_heap_prefix.h"

PAS_BEGIN_EXTERN_C;

#define thingy_try_allocate_primitive __thingy_try_allocate_primitive
#define thingy_try_allocate_primitive_zeroed __thingy_try_allocate_primitive_zeroed
#define thingy_try_allocate_primitive_with_alignment __thingy_try_allocate_primitive_with_alignment

#define thingy_try_reallocate_primitive __thingy_try_reallocate_primitive

#define thingy_try_allocate __thingy_try_allocate
#define thingy_try_allocate_zeroed __thingy_try_allocate_zeroed
#define thingy_try_allocate_array __thingy_try_allocate_array
#define thingy_try_allocate_zeroed_array __thingy_try_allocate_zeroed_array

#define thingy_try_reallocate_array __thingy_try_reallocate_array

#define thingy_deallocate __thingy_deallocate

extern PAS_API pas_heap thingy_primitive_heap;
extern PAS_API pas_heap thingy_utility_heap;
extern PAS_API pas_intrinsic_heap_support thingy_primitive_heap_support;
extern PAS_API pas_intrinsic_heap_support thingy_utility_heap_support;
extern PAS_API pas_allocator_counts thingy_allocator_counts;

PAS_API size_t thingy_get_allocation_size(void*);

PAS_API pas_heap* thingy_heap_ref_get_heap(pas_heap_ref* heap_ref);

PAS_API void* thingy_utility_heap_allocate(size_t size, pas_allocation_mode allocation_mode);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_THINGY */

#endif /* THINGY_HEAP_H */

