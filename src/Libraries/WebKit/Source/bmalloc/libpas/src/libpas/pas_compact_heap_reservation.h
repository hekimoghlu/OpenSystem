/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#ifndef PAS_COMPACT_HEAP_RESERVATION_H
#define PAS_COMPACT_HEAP_RESERVATION_H

#include "pas_aligned_allocation_result.h"
#include "pas_allocation_config.h"
#include "pas_simple_large_free_heap.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern size_t pas_compact_heap_reservation_size;
PAS_API extern size_t pas_compact_heap_reservation_guard_size;
PAS_API extern uintptr_t pas_compact_heap_reservation_base;
PAS_API extern size_t pas_compact_heap_reservation_available_size;
PAS_API extern size_t pas_compact_heap_reservation_bump;

/* FIXME: This should support pas_alignment at some point. */
PAS_API pas_aligned_allocation_result pas_compact_heap_reservation_try_allocate(
    size_t size, size_t alignment);

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_HEAP_RESERVATION_H */
