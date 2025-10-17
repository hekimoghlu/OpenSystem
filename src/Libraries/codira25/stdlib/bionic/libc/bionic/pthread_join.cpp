/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#include <errno.h>

#include "private/bionic_defs.h"
#include "private/bionic_futex.h"
#include "private/bionic_systrace.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_join(pthread_t t, void** return_value) {
  ScopedTrace trace("pthread_join");
  if (t == pthread_self()) {
    return EDEADLK;
  }

  pthread_internal_t* thread = __pthread_internal_find(t, "pthread_join");
  if (thread == nullptr) {
    return ESRCH;
  }

  ThreadJoinState old_state = THREAD_NOT_JOINED;
  while ((old_state == THREAD_NOT_JOINED || old_state == THREAD_EXITED_NOT_JOINED) &&
         !atomic_compare_exchange_weak(&thread->join_state, &old_state, THREAD_JOINED)) {
  }

  if (old_state == THREAD_DETACHED || old_state == THREAD_JOINED) {
    return EINVAL;
  }

  pid_t tid = thread->tid;
  volatile int* tid_ptr = &thread->tid;

  // We set thread->join_state to THREAD_JOINED with atomic operation,
  // so no one is going to remove this thread except us.

  // Wait for the thread to actually exit, if it hasn't already.
  while (*tid_ptr != 0) {
    __futex_wait(tid_ptr, tid, nullptr);
  }

  if (return_value) {
    *return_value = thread->return_value;
  }

  __pthread_internal_remove_and_free(thread);
  return 0;
}
