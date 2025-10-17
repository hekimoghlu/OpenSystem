/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef PAS_THREAD_LOCAL_CACHE_NODE_H
#define PAS_THREAD_LOCAL_CACHE_NODE_H

#include "pas_config.h"
#include "pas_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* Thread local caches have some data that cannot move when the TLC resizes and that should
   stick around in memory even if the TLC dies. It can be reused if some new TLC is born and
   needs that data. That's what the TLC node is for. */

struct pas_thread_local_cache;
struct pas_thread_local_cache_node;
typedef struct pas_thread_local_cache pas_thread_local_cache;
typedef struct pas_thread_local_cache_node pas_thread_local_cache_node;

struct pas_thread_local_cache_node {
    /* Used to track the free nodes. */
    pas_thread_local_cache_node* next_free;
    
    /* Used to track all nodes, free and allocated. */
    pas_thread_local_cache_node* next;
    
    pas_lock page_lock;
    
    pas_lock scavenger_lock;
    
    pas_thread_local_cache* cache;
};

PAS_API extern pas_thread_local_cache_node* pas_thread_local_cache_node_first_free;
PAS_API extern pas_thread_local_cache_node* pas_thread_local_cache_node_first;

/* These functions assume that the heap lock is held. */
PAS_API pas_thread_local_cache_node* pas_thread_local_cache_node_allocate(void);
PAS_API void pas_thread_local_cache_node_deallocate(pas_thread_local_cache_node* node);

PAS_END_EXTERN_C;

#endif /* PAS_THREAD_LOCAL_CACHE_NODE_H */

