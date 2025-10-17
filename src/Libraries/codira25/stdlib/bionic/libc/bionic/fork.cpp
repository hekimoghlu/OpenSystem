/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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

#include <android/fdsan.h>

#include "private/bionic_defs.h"
#include "private/bionic_fdtrack.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE_INLINE
int __clone_for_fork() {
  pthread_internal_t* self = __get_thread();

  int result = clone(nullptr, nullptr, (CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | SIGCHLD),
                     nullptr, nullptr, nullptr, &(self->tid));

  if (result == 0) {
    // Update the cached pid in child, since clone() will not set it directly (as
    // self->tid is updated by the kernel).
    self->set_cached_pid(gettid());
  }

  return result;
}

int _Fork() {
  return __clone_for_fork();
}

int fork() {
  __bionic_atfork_run_prepare();
  int result = _Fork();
  if (result == 0) {
    // Disable fdsan and fdtrack post-fork, so we don't falsely trigger on processes that
    // fork, close all of their fds, and then exec.
    android_fdsan_set_error_level(ANDROID_FDSAN_ERROR_LEVEL_DISABLED);
    android_fdtrack_set_globally_enabled(false);

    // Reset the stack_and_tls VMA name so it doesn't end with a tid from the
    // parent process.
    __set_stack_and_tls_vma_name(true);

    __bionic_atfork_run_child();
  } else {
    __bionic_atfork_run_parent();
  }
  return result;
}
