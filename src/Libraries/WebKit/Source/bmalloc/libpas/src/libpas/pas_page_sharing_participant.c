/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_page_sharing_participant.h"

#include "pas_bitfit_directory.h"
#include "pas_epoch.h"
#include "pas_heap_lock.h"
#include "pas_large_sharing_pool.h"
#include "pas_page_sharing_pool.h"
#include "pas_segregated_size_directory.h"
#include "pas_segregated_shared_page_directory.h"
#include "pas_segregated_size_directory.h"

pas_page_sharing_participant
pas_page_sharing_participant_create(void* ptr,
                                    pas_page_sharing_participant_kind kind)
{
    pas_page_sharing_participant result;
    
    result = (pas_page_sharing_participant)((uintptr_t)ptr | (uintptr_t)kind);
    
    PAS_ASSERT(pas_page_sharing_participant_get_ptr(result) == ptr);
    PAS_ASSERT(pas_page_sharing_participant_get_kind(result) == kind);
    
    return result;
}

pas_page_sharing_participant_payload*
pas_page_sharing_participant_get_payload(pas_page_sharing_participant participant)
{
    static const bool verbose = false;
    
    void* ptr = pas_page_sharing_participant_get_ptr(participant);
    switch (pas_page_sharing_participant_get_kind(participant)) {
    case pas_page_sharing_participant_null:
        PAS_ASSERT(!"Null participant has no payload.");
        return NULL;
    case pas_page_sharing_participant_segregated_shared_page_directory:
    case pas_page_sharing_participant_segregated_size_directory: {
        pas_page_sharing_participant_payload* result;
        if (verbose)
            pas_log("Getting the payload for directory.\n");
        result = pas_segregated_directory_data_try_get_sharing_payload(
            pas_segregated_directory_data_ptr_load(&((pas_segregated_directory*)ptr)->data));
        if (verbose)
            pas_log("Payload = %p\n", result);
        return result;
    }
    case pas_page_sharing_participant_bitfit_directory:
        return &((pas_bitfit_directory*)ptr)->physical_sharing_payload;
    case pas_page_sharing_participant_large_sharing_pool:
        return &pas_large_sharing_participant_payload.base;
    }
    PAS_ASSERT(!"Bad participant kind");
    return NULL;
}

void pas_page_sharing_participant_payload_construct(pas_page_sharing_participant_payload* payload)
{
    *payload = PAS_PAGE_SHARING_PARTICIPANT_PAYLOAD_INITIALIZER;
}

void pas_page_sharing_participant_payload_with_use_epoch_construct(
    pas_page_sharing_participant_payload_with_use_epoch* payload)
{
    *payload = PAS_PAGE_SHARING_PARTICIPANT_PAYLOAD_WITH_USE_EPOCH_INITIALIZER;
}

uint64_t pas_page_sharing_participant_get_use_epoch(pas_page_sharing_participant participant)
{
    void* ptr = pas_page_sharing_participant_get_ptr(participant);
    switch (pas_page_sharing_participant_get_kind(participant)) {
    case pas_page_sharing_participant_null:
        PAS_ASSERT(!"Null participant has no use epoch.");
        return 0;
    case pas_page_sharing_participant_segregated_shared_page_directory:
    case pas_page_sharing_participant_segregated_size_directory:
        return pas_segregated_directory_get_use_epoch(ptr);
    case pas_page_sharing_participant_bitfit_directory:
        return pas_bitfit_directory_get_use_epoch(ptr);
    case pas_page_sharing_participant_large_sharing_pool:
        return pas_large_sharing_participant_payload.use_epoch;
    }
    PAS_ASSERT(!"Bad participant kind");
    return 0;
}

void pas_page_sharing_participant_set_parent_pool(pas_page_sharing_participant participant,
                                                  pas_page_sharing_pool* pool)
{
    PAS_ASSERT(pas_page_sharing_participant_get_parent_pool(participant) == pool);
}

pas_page_sharing_pool*
pas_page_sharing_participant_get_parent_pool(pas_page_sharing_participant participant)
{
    void* ptr;

    ptr = pas_page_sharing_participant_get_ptr(participant);

    PAS_UNUSED_PARAM(ptr);

    switch (pas_page_sharing_participant_get_kind(participant)) {
    case pas_page_sharing_participant_null:
        PAS_ASSERT(!"Cannot get null participant's parent.");
        return NULL;
    case pas_page_sharing_participant_segregated_shared_page_directory:
    case pas_page_sharing_participant_segregated_size_directory:
    case pas_page_sharing_participant_bitfit_directory:
        return &pas_physical_page_sharing_pool;
    case pas_page_sharing_participant_large_sharing_pool:
        return &pas_physical_page_sharing_pool;
    }
    PAS_ASSERT(!"Bad participant kind");
    return NULL;
}

bool pas_page_sharing_participant_is_eligible(pas_page_sharing_participant participant)
{
    void* ptr;

    ptr = pas_page_sharing_participant_get_ptr(participant);

    switch (pas_page_sharing_participant_get_kind(participant)) {
    case pas_page_sharing_participant_null:
        PAS_ASSERT(!"Cannot check if null participant is eligible.");
        return false;
    case pas_page_sharing_participant_segregated_shared_page_directory:
    case pas_page_sharing_participant_segregated_size_directory:
        return !!pas_segregated_directory_get_last_empty_plus_one(ptr).value;
    case pas_page_sharing_participant_bitfit_directory:
        return !!((pas_bitfit_directory*)ptr)->last_empty_plus_one.value;
    case pas_page_sharing_participant_large_sharing_pool:
        return !!pas_large_sharing_min_heap_instance.size;
    }
    PAS_ASSERT(!"Bad participant kind");
    return false;
}

pas_page_sharing_pool_take_result
pas_page_sharing_participant_take_least_recently_used(
    pas_page_sharing_participant participant,
    pas_deferred_decommit_log* decommit_log,
    pas_lock_hold_mode heap_lock_hold_mode)
{
    void* ptr;
    pas_page_sharing_pool_take_result result;

    ptr = pas_page_sharing_participant_get_ptr(participant);

    switch (pas_page_sharing_participant_get_kind(participant)) {
    case pas_page_sharing_participant_null:
        PAS_ASSERT(!"Cannot take from null participant.");
        return false;
        
    case pas_page_sharing_participant_segregated_size_directory:
        return pas_segregated_size_directory_take_last_empty(
            ptr, decommit_log, heap_lock_hold_mode);

    case pas_page_sharing_participant_segregated_shared_page_directory:
        PAS_ASSERT(decommit_log);
        return pas_segregated_shared_page_directory_take_last_empty(
            ptr, decommit_log, heap_lock_hold_mode);

    case pas_page_sharing_participant_bitfit_directory:
        PAS_ASSERT(decommit_log);
        return pas_bitfit_directory_take_last_empty(ptr, decommit_log, heap_lock_hold_mode);
        
    case pas_page_sharing_participant_large_sharing_pool:
        PAS_ASSERT(decommit_log);
        pas_heap_lock_lock_conditionally(heap_lock_hold_mode);
        result = pas_large_sharing_pool_decommit_least_recently_used(decommit_log);
        pas_heap_lock_unlock_conditionally(heap_lock_hold_mode);
        return result;
    }
    PAS_ASSERT(!"Bad participant kind");
    return pas_page_sharing_pool_take_none_available;
}

#endif /* LIBPAS_ENABLED */
