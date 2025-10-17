/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#ifndef PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_H
#define PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_thread_local_cache_layout_node_kind {
    pas_thread_local_cache_layout_segregated_size_directory_node_kind,
    pas_thread_local_cache_layout_redundant_local_allocator_node_kind,
    pas_thread_local_cache_layout_local_view_cache_node_kind
};

typedef enum pas_thread_local_cache_layout_node_kind pas_thread_local_cache_layout_node_kind;
static inline const char* pas_thread_local_cache_layout_node_kind_get_string(
    pas_thread_local_cache_layout_node_kind kind)
{
    switch (kind) {
    case pas_thread_local_cache_layout_segregated_size_directory_node_kind:
        return "segregated_size_directory";
    case pas_thread_local_cache_layout_redundant_local_allocator_node_kind:
        return "redundant_local_allocator";
    case pas_thread_local_cache_layout_local_view_cache_node_kind:
        return "local_view_cache";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_THREAD_LOCAL_CACHE_LAYOUT_NODE_KIND_H */

