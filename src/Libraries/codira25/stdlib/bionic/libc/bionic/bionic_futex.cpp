/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#include "private/bionic_futex.h"

#include <stdatomic.h>
#include <time.h>

#include "private/bionic_time_conversions.h"

static inline __always_inline int FutexWithTimeout(volatile void* ftx, int op, int value,
                                                   bool use_realtime_clock,
                                                   const timespec* abs_timeout, int bitset) {
  // pthread's and semaphore's default behavior is to use CLOCK_REALTIME, however this behavior is
  // essentially never intended, as that clock is prone to change discontinuously.
  //
  // What users really intend is to use CLOCK_MONOTONIC, however only pthread_cond_timedwait()
  // provides this as an option and even there, a large amount of existing code does not opt into
  // CLOCK_MONOTONIC.
  //
  // We have seen numerous bugs directly attributable to this difference.  Therefore, we provide
  // this general workaround to always use CLOCK_MONOTONIC for waiting, regardless of what the input
  // timespec is.
  timespec converted_timeout;
  if (abs_timeout) {
    if ((op & FUTEX_CMD_MASK) == FUTEX_LOCK_PI) {
      if (!use_realtime_clock) {
        realtime_time_from_monotonic_time(converted_timeout, *abs_timeout);
        abs_timeout = &converted_timeout;
      }
    } else {
      op &= ~FUTEX_CLOCK_REALTIME;
      if (use_realtime_clock) {
        monotonic_time_from_realtime_time(converted_timeout, *abs_timeout);
        abs_timeout = &converted_timeout;
      }
    }
    if (abs_timeout->tv_sec < 0) {
      return -ETIMEDOUT;
    }
  }

  return __futex(ftx, op, value, abs_timeout, bitset);
}

int __futex_wait_ex(volatile void* ftx, bool shared, int value, bool use_realtime_clock,
                    const timespec* abs_timeout) {
  return FutexWithTimeout(ftx, (shared ? FUTEX_WAIT_BITSET : FUTEX_WAIT_BITSET_PRIVATE), value,
                          use_realtime_clock, abs_timeout, FUTEX_BITSET_MATCH_ANY);
}

int __futex_pi_lock_ex(volatile void* ftx, bool shared, bool use_realtime_clock,
                       const timespec* abs_timeout) {
  // We really want FUTEX_LOCK_PI2 which is default CLOCK_MONOTONIC, but that isn't supported
  // on linux before 5.14.  FUTEX_LOCK_PI uses CLOCK_REALTIME.  Here we verify support.

  static atomic_int lock_op = 0;
  int op = atomic_load_explicit(&lock_op, memory_order_relaxed);
  if (op == 0) {
    uint32_t tmp = 0;
    if (__futex(&tmp, FUTEX_LOCK_PI2, 0, nullptr, 0) == 0) {
      __futex(&tmp, FUTEX_UNLOCK_PI, 0, nullptr, 0);
      op = FUTEX_LOCK_PI2;
    } else {
      op = FUTEX_LOCK_PI;
    }
    atomic_store_explicit(&lock_op, op, memory_order_relaxed);
  }

  if (!shared) op |= FUTEX_PRIVATE_FLAG;
  return FutexWithTimeout(ftx, op, 0 /* value */, use_realtime_clock, abs_timeout, 0 /* bitset */);
}
