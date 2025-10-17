/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#ifndef PAS_SCAVENGER_H
#define PAS_SCAVENGER_H

#include "pas_utils.h"
#if PAS_OS(DARWIN)
#include <sys/qos.h>
#endif

PAS_BEGIN_EXTERN_C;

#include "pas_allocation_mode.h"

enum pas_scavenger_state {
    pas_scavenger_state_no_thread,
    pas_scavenger_state_polling,
    pas_scavenger_state_deep_sleep
};

typedef enum pas_scavenger_state pas_scavenger_state;

PAS_END_EXTERN_C;

#include <pthread.h>

PAS_BEGIN_EXTERN_C;

struct pas_scavenger_data;
typedef struct pas_scavenger_data pas_scavenger_data;

/* Holds data that needs to be initialized somehow. */
struct pas_scavenger_data {
    pthread_mutex_t lock;
    pthread_cond_t cond;
};

/* This is available extern for testing and debugging only. */
PAS_API extern bool pas_scavenger_is_enabled;
PAS_API extern bool pas_scavenger_eligibility_notification_has_been_deferred;
PAS_API extern pas_scavenger_state pas_scavenger_current_state;
PAS_API extern unsigned pas_scavenger_should_suspend_count;
PAS_API extern pas_scavenger_data* pas_scavenger_data_instance;

/* It's generally not a good idea to mess with this setting, but you an do, at any time. */
PAS_API extern double pas_scavenger_deep_sleep_timeout_in_milliseconds; /* How long to sleep
                                                                           before shutting the
                                                                           thread down. */
PAS_API extern double pas_scavenger_period_in_milliseconds; /* How long to sleep between
                                                               scavenges. */
PAS_API extern uint64_t pas_scavenger_max_epoch_delta; /* How much to subtract from the current epoch
                                                          to compute the max epoch. */

#if PAS_OS(DARWIN)
/* It's legal to set this anytime. */
PAS_API void pas_scavenger_set_requested_qos_class(qos_class_t);
#endif

typedef void (*pas_scavenger_activity_callback)(void);

/* This gets called when the scavenger thread has started. */
PAS_API extern pas_scavenger_activity_callback pas_scavenger_did_start_callback;

/* This gets called anytime the scavenger takes some pages and is about to consider sleeping.
   It's called from the scavenger thread with no locks held. */
PAS_API extern pas_scavenger_activity_callback pas_scavenger_completion_callback;

/* This gets called when the scavenger thread is about to shut down.  It's called from the
   scavenger thread with no locks held. */
PAS_API extern pas_scavenger_activity_callback pas_scavenger_will_shut_down_callback;

/* This defers an eligibility notification. */
PAS_API bool pas_scavenger_did_create_eligible(void);

/* This executes an eligibility notification if there was one. */
PAS_API void pas_scavenger_notify_eligibility_if_needed(void);

PAS_API void pas_scavenger_suspend(void);
PAS_API void pas_scavenger_resume(void);

PAS_API void pas_scavenger_clear_all_non_tlc_caches(void);
PAS_API void pas_scavenger_clear_all_caches_except_remote_tlcs(void);
PAS_API void pas_scavenger_clear_all_caches(void);
PAS_API void pas_scavenger_decommit_expendable_memory(void);
PAS_API void pas_scavenger_fake_decommit_expendable_memory(void); /* Useful for testing. */
PAS_API size_t pas_scavenger_decommit_free_memory(void);

PAS_API void pas_scavenger_run_synchronously_now(void);

typedef enum {
    pas_scavenger_invalid_synchronous_operation_kind,
    pas_scavenger_clear_all_non_tlc_caches_kind,
    pas_scavenger_clear_all_caches_except_remote_tlcs_kind,
    pas_scavenger_clear_all_caches_kind,
    pas_scavenger_decommit_expendable_memory_kind,
    pas_scavenger_decommit_free_memory_kind,
    pas_scavenger_run_synchronously_now_kind
} pas_scavenger_synchronous_operation_kind;

PAS_API void pas_scavenger_perform_synchronous_operation(
    pas_scavenger_synchronous_operation_kind kind);

/* This is for debugging purpose only. */
PAS_API void pas_scavenger_disable_shut_down(void);

PAS_END_EXTERN_C;

#endif /* PAS_SCAVENGER_H */

