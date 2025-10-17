/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#ifndef PAS_COMPACT_EXPENDABLE_MEMORY_H
#define PAS_COMPACT_EXPENDABLE_MEMORY_H

#include "pas_expendable_memory.h"

PAS_BEGIN_EXTERN_C;

#define PAS_COMPACT_EXPENDABLE_MEMORY_PAYLOAD_SIZE 20lu * 1024lu * 1024lu
#define PAS_COMPACT_EXPENDABLE_MEMORY_HEADER_SIZE \
    (PAS_OFFSETOF(pas_expendable_memory, states) + \
     PAS_COMPACT_EXPENDABLE_MEMORY_PAYLOAD_SIZE / PAS_EXPENDABLE_MEMORY_PAGE_SIZE \
     * sizeof(pas_expendable_memory_state))

union pas_compact_expendable_memory;
typedef union pas_compact_expendable_memory pas_compact_expendable_memory;

union pas_compact_expendable_memory {
    pas_expendable_memory header;
    char fake_field_to_force_size[PAS_COMPACT_EXPENDABLE_MEMORY_HEADER_SIZE];
};

PAS_API extern pas_compact_expendable_memory pas_compact_expendable_memory_header;
PAS_API extern void* pas_compact_expendable_memory_payload;

PAS_API void* pas_compact_expendable_memory_allocate(size_t size,
                                                     size_t alignment,
                                                     const char* name);

PAS_API bool pas_compact_expendable_memory_commit_if_necessary(void* object, size_t size);

static inline void pas_compact_expendable_memory_note_use(void* object, size_t size)
{
    pas_expendable_memory_note_use(
        &pas_compact_expendable_memory_header.header, pas_compact_expendable_memory_payload, object, size);
}

static PAS_ALWAYS_INLINE bool pas_compact_expendable_memory_touch(
    void* object, size_t size, pas_expendable_memory_touch_kind kind)
{
    switch (kind) {
    case pas_expendable_memory_touch_to_note_use:
        pas_compact_expendable_memory_note_use(object, size);
        return false;
    case pas_expendable_memory_touch_to_commit_if_necessary:
        return pas_compact_expendable_memory_commit_if_necessary(object, size);
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

PAS_API bool pas_compact_expendable_memory_scavenge(pas_expendable_memory_scavenge_kind kind);

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_EXPENDABLE_MEMORY_H */

