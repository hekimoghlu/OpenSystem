/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include <signal.h>

#include <platform/bionic/reserved_signals.h>

#include "private/SigSetConverter.h"

extern "C" int __rt_sigprocmask(int, const sigset64_t*, sigset64_t*, size_t);

//
// These need to be kept separate from pthread_sigmask, sigblock, sigsetmask,
// sighold, and sigset because libsigchain only intercepts sigprocmask so we
// can't allow clang to decide to inline sigprocmask.
//

int sigprocmask64(int how,
                  const sigset64_t* new_set,
                  sigset64_t* old_set) __attribute__((__noinline__)) {
  // how is only checked for validity if new_set is provided.
  if (new_set && how != SIG_BLOCK && how != SIG_UNBLOCK && how != SIG_SETMASK) {
    errno = EINVAL;
    return -1;
  }

  sigset64_t mutable_new_set;
  sigset64_t* mutable_new_set_ptr = nullptr;
  if (new_set) {
    mutable_new_set = filter_reserved_signals(*new_set, how);
    mutable_new_set_ptr = &mutable_new_set;
  }
  return __rt_sigprocmask(how, mutable_new_set_ptr, old_set, sizeof(*new_set));
}

#if defined(__LP64__)
// For LP64, `sigset64_t` and `sigset_t` are the same.
__strong_alias(sigprocmask, sigprocmask64);
#else
// ILP32 needs a shim.
int sigprocmask(int how,
                const sigset_t* bionic_new_set,
                sigset_t* bionic_old_set) __attribute__((__noinline__)) {
  SigSetConverter new_set{bionic_new_set};
  SigSetConverter old_set{bionic_old_set};
  int rc = sigprocmask64(how, new_set.ptr, old_set.ptr);
  if (rc == 0 && bionic_old_set != nullptr) {
    old_set.copy_out();
  }
  return rc;
}
#endif
