/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#pragma once

#include <sys/cdefs.h>

#include <signal.h>

#include "macros.h"

// Realtime signals reserved for internal use:
//   32 (__SIGRTMIN + 0)        POSIX timers
//   33 (__SIGRTMIN + 1)        libbacktrace
//   34 (__SIGRTMIN + 2)        libcore
//   35 (__SIGRTMIN + 3)        debuggerd
//   36 (__SIGRTMIN + 4)        platform profilers (heapprofd, traced_perf)
//   37 (__SIGRTMIN + 5)        coverage (libprofile-extras)
//   38 (__SIGRTMIN + 6)        heapprofd ART managed heap dumps
//   39 (__SIGRTMIN + 7)        fdtrack
//   40 (__SIGRTMIN + 8)        android_run_on_all_threads (bionic/pthread_internal.cpp)
//   41 (__SIGRTMIN + 9)        re-enable MTE on thread

#define BIONIC_SIGNAL_POSIX_TIMERS (__SIGRTMIN + 0)
#define BIONIC_SIGNAL_BACKTRACE (__SIGRTMIN + 1)
#define BIONIC_SIGNAL_DEBUGGER (__SIGRTMIN + 3)
#define BIONIC_SIGNAL_PROFILER (__SIGRTMIN + 4)
// When used for the dumping a heap dump, BIONIC_SIGNAL_ART_PROFILER is always handled
// gracefully without crashing.
// In debuggerd, we crash the process with this signal to indicate to init that
// a process has been terminated by an MTEAERR SEGV. This works because there is
// no other reason a process could have terminated with this signal.
// This is to work around the limitation of that it is not possible to get the
// si_code that terminated a process.
#define BIONIC_SIGNAL_ART_PROFILER (__SIGRTMIN + 6)
#define BIONIC_SIGNAL_FDTRACK (__SIGRTMIN + 7)
#define BIONIC_SIGNAL_RUN_ON_ALL_THREADS (__SIGRTMIN + 8)
#define BIONIC_ENABLE_MTE (__SIGRTMIN + 9)

#define __SIGRT_RESERVED 10
static inline __always_inline sigset64_t filter_reserved_signals(sigset64_t sigset, int how) {
  int (*block)(sigset64_t*, int);
  int (*unblock)(sigset64_t*, int);
  switch (how) {
    case SIG_BLOCK:
      __BIONIC_FALLTHROUGH;
    case SIG_SETMASK:
      block = sigaddset64;
      unblock = sigdelset64;
      break;

    case SIG_UNBLOCK:
      block = sigdelset64;
      unblock = sigaddset64;
      break;
  }

  // The POSIX timer signal must be blocked.
  block(&sigset, __SIGRTMIN + 0);

  // Everything else must remain unblocked.
  unblock(&sigset, __SIGRTMIN + 1);
  unblock(&sigset, __SIGRTMIN + 2);
  unblock(&sigset, __SIGRTMIN + 3);
  unblock(&sigset, __SIGRTMIN + 4);
  unblock(&sigset, __SIGRTMIN + 5);
  unblock(&sigset, __SIGRTMIN + 6);
  unblock(&sigset, __SIGRTMIN + 7);
  unblock(&sigset, __SIGRTMIN + 8);
  unblock(&sigset, __SIGRTMIN + 9);
  return sigset;
}
