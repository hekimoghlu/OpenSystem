/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#ifndef PAS_PAGE_SHARING_PARTICIPANT_H
#define PAS_PAGE_SHARING_PARTICIPANT_H

#include "pas_config.h"
#include "pas_lock.h"
#include "pas_page_sharing_participant_kind.h"
#include "pas_page_sharing_pool_take_result.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_deferred_decommit_log;
struct pas_page_base;
struct pas_page_base_config;
struct pas_page_sharing_participant_opaque;
struct pas_page_sharing_participant_payload;
struct pas_page_sharing_participant_payload_with_use_epoch;
struct pas_page_sharing_pool;
typedef struct pas_deferred_decommit_log pas_deferred_decommit_log;
typedef struct pas_page_base pas_page_base;
typedef struct pas_page_base_config pas_page_base_config;
typedef struct pas_page_sharing_participant_opaque* pas_page_sharing_participant;
typedef struct pas_page_sharing_participant_payload pas_page_sharing_participant_payload;
typedef struct pas_page_sharing_participant_payload_with_use_epoch pas_page_sharing_participant_payload_with_use_epoch;
typedef struct pas_page_sharing_pool pas_page_sharing_pool;

struct pas_page_sharing_participant_payload {
    uint64_t use_epoch_for_min_heap;
    
    unsigned index_in_sharing_pool;
    unsigned index_in_sharing_pool_min_heap; /* this is a one-based index; will be zero to indicate
                                                that it's not in the pool. */
    
    /* Delta = either the use_epoch changed or a page became available.
    
       If this is false then:
       
       - use_epoch == use_epoch_for_min_heap.
       - it cannot be the case that there are empty pages but we don't belong to the min_heap.
       
       If this is true then:
       
       - we must be tracked as a delta.
    
       FIXME: There's a pretty obvious size win to be had just from removing this field. */
    bool delta_has_been_noted;
};

struct pas_page_sharing_participant_payload_with_use_epoch {
    pas_page_sharing_participant_payload base;
    
    uint64_t use_epoch;
};

#define PAS_PAGE_SHARING_PARTICIPANT_PAYLOAD_INITIALIZER \
    ((pas_page_sharing_participant_payload){ \
        .use_epoch_for_min_heap = 0, \
        .index_in_sharing_pool = 0, \
        .index_in_sharing_pool_min_heap = 0, \
        .delta_has_been_noted = false \
    })

#define PAS_PAGE_SHARING_PARTICIPANT_PAYLOAD_WITH_USE_EPOCH_INITIALIZER \
    ((pas_page_sharing_participant_payload_with_use_epoch){ \
        .base = PAS_PAGE_SHARING_PARTICIPANT_PAYLOAD_INITIALIZER, \
        .use_epoch = 0 \
    })

static inline void* pas_page_sharing_participant_get_ptr(pas_page_sharing_participant participant)
{
    return (void*)((uintptr_t)participant & ~PAS_PAGE_SHARING_PARTICIPANT_KIND_MASK);
}

static inline pas_page_sharing_participant_kind
pas_page_sharing_participant_get_kind(pas_page_sharing_participant participant)
{
    return (pas_page_sharing_participant_kind)(
        (uintptr_t)participant & PAS_PAGE_SHARING_PARTICIPANT_KIND_MASK);
}

PAS_API pas_page_sharing_participant
pas_page_sharing_participant_create(void* ptr,
                                    pas_page_sharing_participant_kind kind);

PAS_API pas_page_sharing_participant_payload*
pas_page_sharing_participant_get_payload(pas_page_sharing_participant participant);

PAS_API void pas_page_sharing_participant_payload_construct(
    pas_page_sharing_participant_payload* payload);

PAS_API void pas_page_sharing_participant_payload_with_use_epoch_construct(
    pas_page_sharing_participant_payload_with_use_epoch* payload);

PAS_API uint64_t pas_page_sharing_participant_get_use_epoch(pas_page_sharing_participant participant);

PAS_API void pas_page_sharing_participant_set_parent_pool(
    pas_page_sharing_participant participant,
    pas_page_sharing_pool* pool);

PAS_API pas_page_sharing_pool*
pas_page_sharing_participant_get_parent_pool(pas_page_sharing_participant participant);

PAS_API bool pas_page_sharing_participant_is_eligible(pas_page_sharing_participant participant);

/* This can return the following take_results:

   - none_available (note this can happen if we previously said is_eligible)
   - locks_unavailable (though not if the decommit_log is empty and we're holding no locks)
   - success */
PAS_API pas_page_sharing_pool_take_result
pas_page_sharing_participant_take_least_recently_used(
    pas_page_sharing_participant participant,
    pas_deferred_decommit_log* decommit_log,
    pas_lock_hold_mode heap_lock_hold_mode);

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_SHARING_PARTICIPANT_H */

