/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#ifndef PAS_IMMORTAL_HEAP_H
#define PAS_IMMORTAL_HEAP_H

#include "pas_allocation_kind.h"
#include "pas_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

extern PAS_API uintptr_t pas_immortal_heap_current;
extern PAS_API uintptr_t pas_immortal_heap_end;
extern PAS_API size_t pas_immortal_heap_allocated_external;
extern PAS_API size_t pas_immortal_heap_allocated_internal;
extern PAS_API size_t pas_immortal_heap_allocation_granule;

PAS_API void* pas_immortal_heap_allocate_with_manual_alignment(size_t size,
                                                               size_t alignment,
                                                               const char* name,
                                                               pas_allocation_kind allocation_kind);

PAS_API void* pas_immortal_heap_allocate_with_alignment(size_t size,
                                                        size_t alignment,
                                                        const char* name,
                                                        pas_allocation_kind allocation_kind);

PAS_API void* pas_immortal_heap_allocate(size_t size,
                                         const char* name,
                                         pas_allocation_kind allocation_kind);

PAS_API void* pas_immortal_heap_hold_lock_and_allocate(size_t size,
                                                       const char* name,
                                                       pas_allocation_kind allocation_kind);

PAS_API void* pas_immortal_heap_allocate_with_heap_lock_hold_mode(
    size_t size,
    const char* name,
    pas_allocation_kind allocation_kind,
    pas_lock_hold_mode heap_lock_hold_mode);

PAS_API void* pas_immortal_heap_allocate_with_alignment_and_heap_lock_hold_mode(
    size_t size,
    size_t alignment,
    const char* name,
    pas_allocation_kind allocation_kind,
    pas_lock_hold_mode heap_lock_hold_mode);

PAS_END_EXTERN_C;

#endif /* PAS_IMMORTAL_HEAP_H */

