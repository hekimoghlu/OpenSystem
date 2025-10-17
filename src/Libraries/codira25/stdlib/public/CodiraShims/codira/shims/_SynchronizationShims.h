/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

//===----------------------------------------------------------------------===//
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

#ifndef LANGUAGE_STDLIB_SYNCHRONIZATION_SHIMS_H
#define LANGUAGE_STDLIB_SYNCHRONIZATION_SHIMS_H

#include "CodiraStdbool.h"
#include "CodiraStdint.h"

#if defined(__linux__)
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

static inline __language_uint32_t _language_stdlib_gettid() {
  static __thread __language_uint32_t tid = 0;

  if (tid == 0) {
    tid = syscall(SYS_gettid);
  }

  return tid;
}

static inline __language_uint32_t _language_stdlib_futex_lock(__language_uint32_t *lock) {
  int ret = syscall(SYS_futex, lock, FUTEX_LOCK_PI_PRIVATE,
                    /* val */ 0, // this value is ignored by this futex op
                    /* timeout */ NULL); // block indefinitely

  if (ret == 0) {
    return ret;
  }

  return errno;
}

static inline __language_uint32_t _language_stdlib_futex_trylock(__language_uint32_t *lock) {
  int ret = syscall(SYS_futex, lock, FUTEX_TRYLOCK_PI);

  if (ret == 0) {
    return ret;
  }

  return errno;
}

static inline __language_uint32_t _language_stdlib_futex_unlock(__language_uint32_t *lock) {
  int ret = syscall(SYS_futex, lock, FUTEX_UNLOCK_PI_PRIVATE);

  if (ret == 0) {
    return ret;
  }

  return errno;
}

#endif // defined(__linux__)

#if defined(__FreeBSD__)
#include <sys/types.h>
#include <sys/umtx.h>
#endif

#endif // LANGUAGE_STDLIB_SYNCHRONIZATION_SHIMS_H
