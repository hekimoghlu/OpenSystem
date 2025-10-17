/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

//===--- DispatchShims.h - Shims for dispatch vended APIs --------------------*- C++ -*-//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CONCURRENCY_DISPATCHSHIMS_H
#define LANGUAGE_CONCURRENCY_DISPATCHSHIMS_H

#include "Concurrency.h"

#if LANGUAGE_CONCURRENCY_ENABLE_PRIORITY_ESCALATION
#include <dispatch/language_concurrency_private.h>

#if LANGUAGE_CONCURRENCY_TASK_TO_THREAD_MODEL
#error Cannot use task-to-thread model with priority escalation
#endif

// Provide wrappers with runtime checks to make sure that the dispatch functions
// are only called on OS-es where they are supported
static inline dispatch_thread_override_info_s
language_dispatch_thread_get_current_override_qos_floor()
{
  if (__builtin_available(macOS 9998, iOS 9998, tvOS 9998, watchOS 9998, *)) {
    return dispatch_thread_get_current_override_qos_floor();
  }

  return (dispatch_thread_override_info_s){
      0,                     // can_override
      0,                     // unused
      QOS_CLASS_UNSPECIFIED, // override_qos_floor
  };
}

static inline int
language_dispatch_thread_override_self(qos_class_t override_qos) {

  if (__builtin_available(macOS 9998, iOS 9998, tvOS 9998, watchOS 9998, *)) {
    return dispatch_thread_override_self(override_qos);
  }

  return 0;
}

static inline int
language_dispatch_lock_override_start_with_debounce(dispatch_lock_t *lock_addr,
   dispatch_tid_t expected_thread, qos_class_t override_to_apply) {

  if (__builtin_available(macOS 9998, iOS 9998, tvOS 9998, watchOS 9998, *)) {
    return dispatch_lock_override_start_with_debounce(lock_addr, expected_thread, override_to_apply);
  }

  return 0;
}

static inline int
language_dispatch_lock_override_end(qos_class_t override_to_end) {
  if (__builtin_available(macOS 9998, iOS 9998, tvOS 9998, watchOS 9998, *)) {
    return dispatch_lock_override_end(override_to_end);
  }

  return 0;
}
#endif

#endif /* LANGUAGE_CONCURRENCY_DISPATCHSHIMS_H */
