/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include <time.h>

#include "private/ErrnoRestorer.h"

int clock_getcpuclockid(pid_t pid, clockid_t* clockid) {
  ErrnoRestorer errno_restorer;

  // The pid is stored in the top bits, but negated.
  clockid_t result = ~static_cast<clockid_t>(pid) << 3;
  // Bits 0 and 1: clock type (0 = CPUCLOCK_PROF, 1 = CPUCLOCK_VIRT, 2 = CPUCLOCK_SCHED).
  result |= 2 /* CPUCLOCK_SCHED */;
  // Bit 2: thread (set) or process (clear).
  result &= ~4 /* CPUCLOCK_PERTHREAD_MASK */;

  if (clock_getres(result, nullptr) == -1) {
    return ESRCH;
  }

  *clockid = result;
  return 0;
}
