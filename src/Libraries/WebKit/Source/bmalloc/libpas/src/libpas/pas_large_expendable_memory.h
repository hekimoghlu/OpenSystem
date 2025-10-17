/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#ifndef PAS_LARGE_EXPENDABLE_MEMORY_H
#define PAS_LARGE_EXPENDABLE_MEMORY_H

#include "pas_expendable_memory.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_expendable_memory;
typedef struct pas_large_expendable_memory pas_large_expendable_memory;

struct pas_large_expendable_memory {
    pas_large_expendable_memory* next;
    pas_expendable_memory header;
};

#define PAS_LARGE_EXPENDABLE_MEMORY_BASE_HEADER_SIZE PAS_OFFSETOF(pas_large_expendable_memory, header.states)
#define PAS_LARGE_EXPENDABLE_MEMORY_HEADER_SIZE PAS_EXPENDABLE_MEMORY_PAGE_SIZE

#define PAS_LARGE_EXPENDABLE_MEMORY_ALIGNMENT \
    (PAS_LARGE_EXPENDABLE_MEMORY_HEADER_SIZE / sizeof(pas_expendable_memory_state) * \
     PAS_EXPENDABLE_MEMORY_PAGE_SIZE)
#define PAS_LARGE_EXPENDABLE_MEMORY_PAYLOAD_SIZE \
    (((PAS_LARGE_EXPENDABLE_MEMORY_HEADER_SIZE - PAS_LARGE_EXPENDABLE_MEMORY_BASE_HEADER_SIZE) / \
      sizeof(pas_expendable_memory_state)) \
     * PAS_EXPENDABLE_MEMORY_PAGE_SIZE)
#define PAS_LARGE_EXPENDABLE_MEMORY_TOTAL_SIZE \
    (PAS_LARGE_EXPENDABLE_MEMORY_HEADER_SIZE + PAS_LARGE_EXPENDABLE_MEMORY_PAYLOAD_SIZE)

#if PAS_COMPILER(CLANG)
_Static_assert(PAS_LARGE_EXPENDABLE_MEMORY_ALIGNMENT > PAS_LARGE_EXPENDABLE_MEMORY_PAYLOAD_SIZE,
               "Large expendable memory should be aligned more so than the payload size.");
_Static_assert(PAS_LARGE_EXPENDABLE_MEMORY_ALIGNMENT > PAS_LARGE_EXPENDABLE_MEMORY_TOTAL_SIZE,
               "Large expendable memory should be aligned more so than the total size.");
#endif

PAS_API extern pas_large_expendable_memory* pas_large_expendable_memory_head;

static inline void* pas_large_expendable_memory_payload(pas_large_expendable_memory* header)
{
    return (char*)header + PAS_LARGE_EXPENDABLE_MEMORY_HEADER_SIZE;
}

static inline pas_large_expendable_memory* pas_large_expendable_memory_header_for_object(void* object)
{
    return (pas_large_expendable_memory*)
        pas_round_down_to_power_of_2((uintptr_t)object, PAS_LARGE_EXPENDABLE_MEMORY_ALIGNMENT);
}

PAS_API void* pas_large_expendable_memory_allocate(size_t size, size_t alignment, const char* name);

PAS_API bool pas_large_expendable_memory_commit_if_necessary(void* object, size_t size);

static inline void pas_large_expendable_memory_note_use(void* object, size_t size)
{
    pas_large_expendable_memory* header;
    void* payload;

    header = pas_large_expendable_memory_header_for_object(object);
    payload = pas_large_expendable_memory_payload(header);
    
    pas_expendable_memory_note_use(&header->header, payload, object, size);
}

static PAS_ALWAYS_INLINE bool pas_large_expendable_memory_touch(
    void* object, size_t size, pas_expendable_memory_touch_kind kind)
{
    switch (kind) {
    case pas_expendable_memory_touch_to_note_use:
        pas_large_expendable_memory_note_use(object, size);
        return false;
    case pas_expendable_memory_touch_to_commit_if_necessary:
        return pas_large_expendable_memory_commit_if_necessary(object, size);
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

PAS_API bool pas_large_expendable_memory_scavenge(pas_expendable_memory_scavenge_kind kind);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_EXPENDABLE_MEMORY_H */

