/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "private/bionic_defs.h"
#include "private/ErrnoRestorer.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_sigqueue(pthread_t t, int sig, const union sigval value) {
  ErrnoRestorer errno_restorer;

  pid_t pid = getpid();

  pid_t tid = __pthread_internal_gettid(t, "pthread_sigqueue");
  if (tid == -1) return ESRCH;

  siginfo_t siginfo = { .si_code = SI_QUEUE };
  siginfo.si_signo = sig;
  siginfo.si_pid = pid;
  siginfo.si_uid = getuid();
  siginfo.si_value = value;

  return syscall(__NR_rt_tgsigqueueinfo, pid, tid, sig, &siginfo) ? errno : 0;
}
