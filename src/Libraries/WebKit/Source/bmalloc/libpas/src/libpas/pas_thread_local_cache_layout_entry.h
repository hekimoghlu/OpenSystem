/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef PAS_THREAD_LOCAL_CACHE_LAYOUT_ENTRY_H
#define PAS_THREAD_LOCAL_CACHE_LAYOUT_ENTRY_H

#include "pas_allocator_index.h"
#include "pas_compact_thread_local_cache_layout_node.h"

PAS_BEGIN_EXTERN_C;

typedef pas_compact_thread_local_cache_layout_node pas_thread_local_cache_layout_entry;
typedef pas_allocator_index pas_thread_local_cache_layout_key;

static inline pas_thread_local_cache_layout_entry pas_thread_local_cache_layout_entry_create_empty(void)
{
    pas_compact_thread_local_cache_layout_node result;
    pas_compact_thread_local_cache_layout_node_store(&result, NULL);
    return result;
}

static inline pas_thread_local_cache_layout_entry pas_thread_local_cache_layout_entry_create_deleted(void)
{
    pas_compact_thread_local_cache_layout_node result;
    pas_compact_thread_local_cache_layout_node_store(&result, (pas_thread_local_cache_layout_node)(uintptr_t)1);
    return result;
}

static inline bool pas_thread_local_cache_layout_entry_is_empty_or_deleted(
    pas_thread_local_cache_layout_entry entry)
{
    return (uintptr_t)pas_compact_thread_local_cache_layout_node_load(&entry) <= (uintptr_t)1;
}

static inline bool pas_thread_local_cache_layout_entry_is_empty(pas_thread_local_cache_layout_entry entry)
{
    return pas_compact_thread_local_cache_layout_node_is_null(&entry);
}

static inline bool pas_thread_local_cache_layout_entry_is_deleted(pas_thread_local_cache_layout_entry entry)
{
    return (uintptr_t)pas_compact_thread_local_cache_layout_node_load(&entry) == (uintptr_t)1;
}

static inline pas_allocator_index pas_thread_local_cache_layout_entry_get_key(
    pas_thread_local_cache_layout_entry entry)
{
    return pas_thread_local_cache_layout_node_get_allocator_index_generic(
        pas_compact_thread_local_cache_layout_node_load_non_null(&entry));
}

static inline unsigned pas_thread_local_cache_layout_key_get_hash(pas_thread_local_cache_layout_key key)
{
    return pas_hash32(key);
}

static inline bool pas_thread_local_cache_layout_key_is_equal(pas_thread_local_cache_layout_key a,
                                                              pas_thread_local_cache_layout_key b)
{
    return a == b;
}

PAS_END_EXTERN_C;

#endif /* PAS_THREAD_LOCAL_CACHE_LAYOUT_ENTRY_H */

