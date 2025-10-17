/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include <unistd.h>

#include "pthread_internal.h"

extern "C" pid_t __getpid();

pid_t __get_cached_pid() {
  pthread_internal_t* self = __get_thread();
  if (__predict_true(self)) {
    pid_t cached_pid;
    if (__predict_true(self->get_cached_pid(&cached_pid))) {
      return cached_pid;
    }
  }
  return 0;
}

pid_t getpid() {
  pid_t cached_pid = __get_cached_pid();
  if (__predict_true(cached_pid != 0)) {
    return cached_pid;
  }

  // We're still in the dynamic linker or we're in the middle of forking, so ask the kernel.
  // We don't know whether it's safe to update the cached value, so don't try.
  return __getpid();
}
