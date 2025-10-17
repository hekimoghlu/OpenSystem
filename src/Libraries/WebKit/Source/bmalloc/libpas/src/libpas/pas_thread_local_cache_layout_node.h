/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#ifndef PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_H
#define PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_H

#include "pas_allocator_index.h"
#include "pas_lock.h"
#include "pas_range.h"
#include "pas_thread_local_cache_layout_node_kind.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_redundant_local_allocator_node;
struct pas_segregated_size_directory;
struct pas_thread_local_cache;
struct pas_thread_local_cache_layout_node_opaque;
typedef struct pas_redundant_local_allocator_node pas_redundant_local_allocator_node;
typedef struct pas_segregated_size_directory pas_segregated_size_directory;
typedef struct pas_thread_local_cache pas_thread_local_cache;
typedef struct pas_thread_local_cache_layout_node_opaque* pas_thread_local_cache_layout_node;

#define PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_MASK 3lu

static inline void* pas_thread_local_cache_layout_node_get_ptr(pas_thread_local_cache_layout_node node)
{
    return (void*)((uintptr_t)node & ~PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_MASK);
}

static inline pas_thread_local_cache_layout_node_kind
pas_thread_local_cache_layout_node_get_kind(pas_thread_local_cache_layout_node node)
{
    return (pas_thread_local_cache_layout_node_kind)(
        (uintptr_t)node & PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_MASK);
}

static inline pas_thread_local_cache_layout_node
pas_thread_local_cache_layout_node_create(pas_thread_local_cache_layout_node_kind kind, void* ptr)
{
    pas_thread_local_cache_layout_node result;
    
    PAS_ASSERT(ptr);

    result = (pas_thread_local_cache_layout_node)((uintptr_t)ptr | (uintptr_t)kind);
    PAS_ASSERT(pas_thread_local_cache_layout_node_get_ptr(result) == ptr);
    PAS_ASSERT(pas_thread_local_cache_layout_node_get_kind(result) == kind);

    return result;
}

static inline pas_thread_local_cache_layout_node
pas_wrap_segregated_size_directory(pas_segregated_size_directory* directory)
{
    return pas_thread_local_cache_layout_node_create(
        pas_thread_local_cache_layout_segregated_size_directory_node_kind, directory);
}

static inline pas_thread_local_cache_layout_node
pas_wrap_redundant_local_allocator_node(pas_redundant_local_allocator_node* node)
{
    return pas_thread_local_cache_layout_node_create(
        pas_thread_local_cache_layout_redundant_local_allocator_node_kind, node);
}

static inline pas_thread_local_cache_layout_node
pas_wrap_local_view_cache_node(pas_segregated_size_directory* directory)
{
    return pas_thread_local_cache_layout_node_create(
        pas_thread_local_cache_layout_local_view_cache_node_kind, directory);
}

static inline bool
pas_is_wrapped_segregated_size_directory(pas_thread_local_cache_layout_node node)
{
    return pas_thread_local_cache_layout_node_get_kind(node)
        == pas_thread_local_cache_layout_segregated_size_directory_node_kind;
}

static inline bool
pas_is_wrapped_redundant_local_allocator_node(pas_thread_local_cache_layout_node node)
{
    return pas_thread_local_cache_layout_node_get_kind(node)
        == pas_thread_local_cache_layout_redundant_local_allocator_node_kind;
}

static inline bool
pas_is_wrapped_local_view_cache_node(pas_thread_local_cache_layout_node node)
{
    return pas_thread_local_cache_layout_node_get_kind(node)
        == pas_thread_local_cache_layout_local_view_cache_node_kind;
}

static inline pas_segregated_size_directory*
pas_unwrap_segregated_size_directory(pas_thread_local_cache_layout_node node)
{
    PAS_ASSERT(pas_is_wrapped_segregated_size_directory(node));
    return (pas_segregated_size_directory*)pas_thread_local_cache_layout_node_get_ptr(node);
}

static inline pas_redundant_local_allocator_node*
pas_unwrap_redundant_local_allocator_node(pas_thread_local_cache_layout_node node)
{
    PAS_ASSERT(pas_is_wrapped_redundant_local_allocator_node(node));
    return (pas_redundant_local_allocator_node*)pas_thread_local_cache_layout_node_get_ptr(node);
}

static inline pas_segregated_size_directory*
pas_unwrap_local_view_cache_node(pas_thread_local_cache_layout_node node)
{
    PAS_ASSERT(pas_is_wrapped_local_view_cache_node(node));
    return (pas_segregated_size_directory*)pas_thread_local_cache_layout_node_get_ptr(node);
}

PAS_API pas_segregated_size_directory*
pas_thread_local_cache_layout_node_get_directory(pas_thread_local_cache_layout_node node);

PAS_API pas_allocator_index
pas_thread_local_cache_layout_node_num_allocator_indices(pas_thread_local_cache_layout_node node);

static inline bool
pas_thread_local_cache_layout_node_represents_allocator(pas_thread_local_cache_layout_node node)
{
    return pas_is_wrapped_segregated_size_directory(node)
        || pas_is_wrapped_redundant_local_allocator_node(node);
}

static inline bool
pas_thread_local_cache_layout_node_represents_view_cache(pas_thread_local_cache_layout_node node)
{
    return pas_is_wrapped_local_view_cache_node(node);
}

PAS_API pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_generic(pas_thread_local_cache_layout_node node);

PAS_API pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_for_allocator(
    pas_thread_local_cache_layout_node node);

PAS_API pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_for_view_cache(
    pas_thread_local_cache_layout_node node);

PAS_API void
pas_thread_local_cache_layout_node_set_allocator_index(pas_thread_local_cache_layout_node node,
                                                       pas_allocator_index index);

PAS_API void pas_thread_local_cache_layout_node_commit_and_construct(pas_thread_local_cache_layout_node node,
                                                                     pas_thread_local_cache* cache);

PAS_API void pas_thread_local_cache_layout_node_move(pas_thread_local_cache_layout_node node,
                                                     pas_thread_local_cache* to_cache,
                                                     pas_thread_local_cache* from_cache);

PAS_API void pas_thread_local_cache_layout_node_stop(pas_thread_local_cache_layout_node node,
                                                     pas_thread_local_cache* cache,
                                                     pas_lock_lock_mode page_lock_mode,
                                                     pas_lock_hold_mode heap_lock_hold_mode);

PAS_API bool pas_thread_local_cache_layout_node_is_committed(pas_thread_local_cache_layout_node node,
                                                             pas_thread_local_cache* cache);
PAS_API void pas_thread_local_cache_layout_node_ensure_committed(pas_thread_local_cache_layout_node node,
                                                                 pas_thread_local_cache* cache);
PAS_API void pas_thread_local_cache_layout_node_prepare_to_decommit(pas_thread_local_cache_layout_node node,
                                                                    pas_thread_local_cache* cache,
                                                                    pas_range decommit_range);

PAS_END_EXTERN_C;

#endif /* PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_H */

