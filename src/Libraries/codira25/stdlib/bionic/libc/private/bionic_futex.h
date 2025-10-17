/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#ifndef _BIONIC_FUTEX_H
#define _BIONIC_FUTEX_H

#include <errno.h>
#include <linux/futex.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/cdefs.h>
#include <sys/syscall.h>
#include <unistd.h>

struct timespec;

static inline __always_inline int __futex(volatile void* ftx, int op, int value,
                                          const timespec* timeout, int bitset) {
  // Our generated syscall assembler sets errno, but our callers (pthread functions) don't want to.
  int saved_errno = errno;
  int result = syscall(__NR_futex, ftx, op, value, timeout, NULL, bitset);
  if (__predict_false(result == -1)) {
    result = -errno;
    errno = saved_errno;
  }
  return result;
}

static inline int __futex_wake(volatile void* ftx, int count) {
  return __futex(ftx, FUTEX_WAKE, count, nullptr, 0);
}

static inline int __futex_wake_ex(volatile void* ftx, bool shared, int count) {
  return __futex(ftx, shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE, count, nullptr, 0);
}

static inline int __futex_wait(volatile void* ftx, int value, const timespec* timeout) {
  return __futex(ftx, FUTEX_WAIT, value, timeout, 0);
}

static inline int __futex_wait_ex(volatile void* ftx, bool shared, int value) {
  return __futex(ftx, (shared ? FUTEX_WAIT_BITSET : FUTEX_WAIT_BITSET_PRIVATE), value, nullptr,
                 FUTEX_BITSET_MATCH_ANY);
}

__LIBC_HIDDEN__ int __futex_wait_ex(volatile void* ftx, bool shared, int value,
                                    bool use_realtime_clock, const timespec* abs_timeout);

static inline int __futex_pi_unlock(volatile void* ftx, bool shared) {
  return __futex(ftx, shared ? FUTEX_UNLOCK_PI : FUTEX_UNLOCK_PI_PRIVATE, 0, nullptr, 0);
}

__LIBC_HIDDEN__ int __futex_pi_lock_ex(volatile void* ftx, bool shared, bool use_realtime_clock,
                                       const timespec* abs_timeout);

#endif /* _BIONIC_FUTEX_H */
