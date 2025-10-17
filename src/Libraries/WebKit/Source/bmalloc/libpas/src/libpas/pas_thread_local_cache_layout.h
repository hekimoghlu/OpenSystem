/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#ifndef PAS_THREAD_LOCAL_CACHE_LAYOUT_H
#define PAS_THREAD_LOCAL_CACHE_LAYOUT_H

#include "pas_allocator_index.h"
#include "pas_compact_atomic_thread_local_cache_layout_node.h"
#include "pas_config.h"
#include "pas_hashtable.h"
#include "pas_lock.h"
#include "pas_thread_local_cache_layout_entry.h"
#include "pas_thread_local_cache_layout_node.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_THREAD_LOCAL_CACHE_LAYOUT_SEGMENT_SIZE 257
struct pas_thread_local_cache_layout_segment {
    pas_compact_atomic_thread_local_cache_layout_node nodes[PAS_THREAD_LOCAL_CACHE_LAYOUT_SEGMENT_SIZE];
    pas_compact_atomic_thread_local_cache_layout_node sentinel;
    struct pas_thread_local_cache_layout_segment* next;
};
typedef struct pas_thread_local_cache_layout_segment pas_thread_local_cache_layout_segment;

PAS_API extern pas_thread_local_cache_layout_segment* pas_thread_local_cache_layout_first_segment;

PAS_CREATE_HASHTABLE(pas_thread_local_cache_layout_hashtable,
                     pas_thread_local_cache_layout_entry,
                     pas_thread_local_cache_layout_key);

/* Lock used internally for accessing the hashtable_instance. You don't have to use this lock unless you
   access the hashtable_instance directly. */
PAS_DECLARE_LOCK(pas_thread_local_cache_layout_hashtable);

PAS_API extern pas_thread_local_cache_layout_hashtable pas_thread_local_cache_layout_hashtable_instance;

/* Clients can use this to force the next call to add to go to this index. */
PAS_API extern pas_allocator_index pas_thread_local_cache_layout_next_allocator_index;

PAS_API pas_allocator_index pas_thread_local_cache_layout_add_node(
    pas_thread_local_cache_layout_node node);

PAS_API pas_allocator_index pas_thread_local_cache_layout_add(
    pas_segregated_size_directory* directory);

PAS_API pas_allocator_index pas_thread_local_cache_layout_duplicate(
    pas_segregated_size_directory* directory);

PAS_API pas_allocator_index pas_thread_local_cache_layout_add_view_cache(
    pas_segregated_size_directory* directory);

/* You don't need to hold any locks to use this because this uses its own lock behind the scenes. */
PAS_API pas_thread_local_cache_layout_node pas_thread_local_cache_layout_get_node_for_index(
    pas_allocator_index index);

PAS_API pas_thread_local_cache_layout_node pas_thread_local_cache_layout_get_last_node(void);

static PAS_ALWAYS_INLINE pas_thread_local_cache_layout_node pas_thread_local_cache_layout_segment_get_node(pas_thread_local_cache_layout_segment* segment, uintptr_t index)
{
    if (!segment)
        return NULL;
    return pas_compact_atomic_thread_local_cache_layout_node_load(&segment->nodes[index]);
}

static PAS_ALWAYS_INLINE pas_thread_local_cache_layout_node pas_thread_local_cache_layout_segment_next_node(pas_thread_local_cache_layout_segment** segment, uintptr_t* index)
{
    PAS_TESTING_ASSERT(segment);
    PAS_TESTING_ASSERT(*segment);
    uintptr_t next_index = *index + 1;
    pas_thread_local_cache_layout_node next = pas_thread_local_cache_layout_segment_get_node(*segment, next_index);
    if (next) {
        *index = next_index;
        return next;
    }
    *segment = (*segment)->next;
    *index = 0;
    return pas_thread_local_cache_layout_segment_get_node(*segment, 0);
}

#define PAS_THREAD_LOCAL_CACHE_LAYOUT_EACH_ALLOCATOR(node) \
    uintptr_t internal_node_index = 0, internal_segment = (uintptr_t)pas_thread_local_cache_layout_first_segment, internal_node = (uintptr_t)pas_thread_local_cache_layout_segment_get_node((pas_thread_local_cache_layout_segment*)internal_segment, internal_node_index); \
    (node = ((pas_thread_local_cache_layout_node)internal_node)); \
    internal_node = (uintptr_t)pas_thread_local_cache_layout_segment_next_node((pas_thread_local_cache_layout_segment**)&internal_segment, &internal_node_index)

#define PAS_THREAD_LOCAL_CACHE_LAYOUT_EACH_ALLOCATOR_WITH_SEGMENT_AND_INDEX(node, segment, index) \
    node = pas_thread_local_cache_layout_segment_get_node(segment, index); \
    node; \
    node = pas_thread_local_cache_layout_segment_next_node(&segment, &index)

PAS_END_EXTERN_C;

#endif /* PAS_THREAD_LOCAL_CACHE_LAYOUT_H */

