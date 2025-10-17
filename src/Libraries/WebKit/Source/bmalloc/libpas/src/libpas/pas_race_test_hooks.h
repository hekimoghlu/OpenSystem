/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
#ifndef PAS_RACE_TEST_HOOKS_H
#define PAS_RACE_TEST_HOOKS_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_race_test_hook_kind {
    pas_race_test_hook_local_allocator_stop_before_did_stop_allocating,
    pas_race_test_hook_local_allocator_stop_before_unlock
};

typedef enum pas_race_test_hook_kind pas_race_test_hook_kind;

static inline const char* pas_race_test_hook_kind_get_string(pas_race_test_hook_kind kind)
{
    switch (kind) {
    case pas_race_test_hook_local_allocator_stop_before_did_stop_allocating:
        return "local_allocator_stop_before_did_stop_allocating";
    case pas_race_test_hook_local_allocator_stop_before_unlock:
        return "local_allocator_stop_before_unlock";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

struct pas_lock;
typedef struct pas_lock pas_lock;

#if PAS_ENABLE_TESTING
typedef void (*pas_race_test_hook_callback)(pas_race_test_hook_kind kind);
typedef void (*pas_race_test_lock_callback)(pas_lock* lock);

PAS_API extern pas_race_test_hook_callback pas_race_test_hook_callback_instance;
PAS_API extern pas_race_test_lock_callback pas_race_test_will_lock_callback;
PAS_API extern pas_race_test_lock_callback pas_race_test_did_lock_callback;
PAS_API extern pas_race_test_lock_callback pas_race_test_did_try_lock_callback;
PAS_API extern pas_race_test_lock_callback pas_race_test_will_unlock_callback;

static inline void pas_race_test_hook(pas_race_test_hook_kind kind)
{
    pas_race_test_hook_callback callback;
    callback = pas_race_test_hook_callback_instance;
    if (callback)
        callback(kind);
}

static inline void pas_race_test_will_lock(pas_lock* lock)
{
    pas_race_test_lock_callback callback;
    callback = pas_race_test_will_lock_callback;
    if (callback)
        callback(lock);
}

static inline void pas_race_test_did_lock(pas_lock* lock)
{
    pas_race_test_lock_callback callback;
    callback = pas_race_test_did_lock_callback;
    if (callback)
        callback(lock);
}

static inline void pas_race_test_did_try_lock(pas_lock* lock)
{
    pas_race_test_lock_callback callback;
    callback = pas_race_test_did_try_lock_callback;
    if (callback)
        callback(lock);
}

static inline void pas_race_test_will_unlock(pas_lock* lock)
{
    pas_race_test_lock_callback callback;
    callback = pas_race_test_will_unlock_callback;
    if (callback)
        callback(lock);
}
#else /* PAS_ENABLE_TESTING -> so !PAS_ENABLE_TESTING */
static inline void pas_race_test_hook(pas_race_test_hook_kind kind) { PAS_UNUSED_PARAM(kind); }
static inline void pas_race_test_will_lock(pas_lock* lock) { PAS_UNUSED_PARAM(lock); }
static inline void pas_race_test_did_lock(pas_lock* lock) { PAS_UNUSED_PARAM(lock); }
static inline void pas_race_test_did_try_lock(pas_lock* lock) { PAS_UNUSED_PARAM(lock); }
static inline void pas_race_test_will_unlock(pas_lock* lock) { PAS_UNUSED_PARAM(lock); }
#endif /* PAS_ENABLE_TESTING -> so end of !PAS_ENABLE_TESTING */

PAS_END_EXTERN_C;

#endif /* PAS_RACE_TEST_HOOKS_H */

