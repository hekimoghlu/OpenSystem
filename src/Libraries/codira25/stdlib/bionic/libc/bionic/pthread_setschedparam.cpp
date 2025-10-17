/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#include <pthread.h>
#include <sched.h>

#include "private/bionic_defs.h"
#include "private/ErrnoRestorer.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_setschedparam(pthread_t t, int policy, const sched_param* param) {
  ErrnoRestorer errno_restorer;

  pid_t tid = __pthread_internal_gettid(t, "pthread_setschedparam");
  if (tid == -1) return ESRCH;

  return (sched_setscheduler(tid, policy, param) == -1) ? errno : 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_setschedprio(pthread_t t, int priority) {
  ErrnoRestorer errno_restorer;

  pid_t tid = __pthread_internal_gettid(t, "pthread_setschedprio");
  if (tid == -1) return ESRCH;

  sched_param param = { .sched_priority = priority };
  return (sched_setparam(tid, &param) == -1) ? errno : 0;
}
