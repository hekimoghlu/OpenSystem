/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#ifndef MINALIGN32_HEAP_H
#define MINALIGN32_HEAP_H

#include "pas_config.h"

#if PAS_ENABLE_MINALIGN32

#include "iso_heap.h"
#include "pas_intrinsic_heap_support.h"
#include "pas_allocation_mode.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_heap minalign32_common_primitive_heap;
PAS_API extern pas_intrinsic_heap_support minalign32_common_primitive_heap_support;

PAS_API void* minalign32_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* minalign32_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode);
PAS_API void* minalign32_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void minalign32_deallocate(void* ptr);
PAS_API pas_heap* minalign32_heap_ref_get_heap(pas_heap_ref* heap_ref);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_MINALIGN32 */

#endif /* MINALIGN32_HEAP_H */

