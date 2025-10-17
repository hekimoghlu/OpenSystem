/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <sys/types.h>

#include "header_checks.h"

static void sys_types_h() {
  TYPE(blkcnt_t);
  TYPE(blksize_t);
  TYPE(clock_t);
  TYPE(clockid_t);
  TYPE(dev_t);
  TYPE(fsblkcnt_t);
  TYPE(fsfilcnt_t);
  TYPE(gid_t);
  TYPE(id_t);
  TYPE(ino_t);
  TYPE(key_t);
  TYPE(mode_t);
  TYPE(nlink_t);
  TYPE(off_t);
  TYPE(pid_t);
  TYPE(pthread_attr_t);
  TYPE(pthread_barrier_t);
  TYPE(pthread_barrierattr_t);
  TYPE(pthread_cond_t);
  TYPE(pthread_condattr_t);
  TYPE(pthread_key_t);
  TYPE(pthread_mutex_t);
  TYPE(pthread_mutexattr_t);
  TYPE(pthread_once_t);
  TYPE(pthread_rwlock_t);
  TYPE(pthread_rwlockattr_t);
  TYPE(pthread_spinlock_t);
  TYPE(pthread_t);
  TYPE(size_t);
  TYPE(ssize_t);
  TYPE(suseconds_t);
  TYPE(time_t);
  TYPE(timer_t);
  TYPE(uid_t);
}
