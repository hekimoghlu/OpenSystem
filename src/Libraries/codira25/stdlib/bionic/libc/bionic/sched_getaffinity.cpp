/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#define _GNU_SOURCE 1
#include <sched.h>
#include <string.h>

extern "C" int __sched_getaffinity(pid_t, size_t, cpu_set_t*);

int sched_getaffinity(pid_t pid, size_t set_size, cpu_set_t* set) {
  int rc = __sched_getaffinity(pid, set_size, set);
  if (rc == -1) {
    return -1;
  }

  // Clear any bytes the kernel didn't touch.
  // (The kernel returns the number of bytes written on success.)
  memset(reinterpret_cast<char*>(set) + rc, 0, set_size - rc);
  return 0;
}
