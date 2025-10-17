/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include <signal.h>
#include <string.h>

#include <platform/bionic/reserved_signals.h>

extern "C" void __restore_rt(void);
extern "C" void __restore(void);

#if defined(__LP64__)

extern "C" int __rt_sigaction(int, const struct __kernel_sigaction*, struct __kernel_sigaction*, size_t);

int sigaction(int signal, const struct sigaction* bionic_new_action, struct sigaction* bionic_old_action) {
  __kernel_sigaction kernel_new_action = {};
  if (bionic_new_action != nullptr) {
    kernel_new_action.sa_flags = bionic_new_action->sa_flags;
    kernel_new_action.sa_handler = bionic_new_action->sa_handler;
    // Don't filter signals here; if the caller asked for everything to be blocked, we should obey.
    kernel_new_action.sa_mask = bionic_new_action->sa_mask;
#if defined(__x86_64__)
    // riscv64 doesn't have sa_restorer. For arm64 and 32-bit x86, unwinding
    // works best if you just let the kernel supply the default restorer
    // from [vdso]. gdb doesn't care, but libgcc needs the nop that the
    // kernel includes before the actual code. (We could add that ourselves,
    // but why bother?)
    // TODO: why do arm32 and x86-64 need this to unwind through signal handlers?
    kernel_new_action.sa_restorer = bionic_new_action->sa_restorer;
    if (!(kernel_new_action.sa_flags & SA_RESTORER)) {
      kernel_new_action.sa_flags |= SA_RESTORER;
      kernel_new_action.sa_restorer = &__restore_rt;
    }
#endif
  }

  __kernel_sigaction kernel_old_action;
  int result = __rt_sigaction(signal,
                              (bionic_new_action != nullptr) ? &kernel_new_action : nullptr,
                              (bionic_old_action != nullptr) ? &kernel_old_action : nullptr,
                              sizeof(sigset_t));

  if (bionic_old_action != nullptr) {
    bionic_old_action->sa_flags = kernel_old_action.sa_flags;
    bionic_old_action->sa_handler = kernel_old_action.sa_handler;
    bionic_old_action->sa_mask = kernel_old_action.sa_mask;
#if defined(SA_RESTORER)
    bionic_old_action->sa_restorer = kernel_old_action.sa_restorer;
#endif
  }

  return result;
}

__strong_alias(sigaction64, sigaction);

#else

extern "C" int __rt_sigaction(int, const struct sigaction64*, struct sigaction64*, size_t);

// sigaction and sigaction64 get interposed in ART: ensure that we don't end up calling
//     sigchain sigaction -> bionic sigaction -> sigchain sigaction64 -> bionic sigaction64
// by extracting the implementation of sigaction64 to a static function.
static int __sigaction64(int signal, const struct sigaction64* bionic_new,
                         struct sigaction64* bionic_old) {
  struct sigaction64 kernel_new = {};
  if (bionic_new) {
    kernel_new = *bionic_new;
#if defined(__arm__)
    // (See sa_restorer comment in sigaction() above.)
    if (!(kernel_new.sa_flags & SA_RESTORER)) {
      kernel_new.sa_flags |= SA_RESTORER;
      kernel_new.sa_restorer = (kernel_new.sa_flags & SA_SIGINFO) ? &__restore_rt : &__restore;
    }
#endif
    // Don't filter signals here; if the caller asked for everything to be blocked, we should obey.
    kernel_new.sa_mask = kernel_new.sa_mask;
  }

  return __rt_sigaction(signal, bionic_new ? &kernel_new : nullptr, bionic_old,
                        sizeof(kernel_new.sa_mask));
}

int sigaction(int signal, const struct sigaction* bionic_new, struct sigaction* bionic_old) {
  // The 32-bit ABI is broken. struct sigaction includes a too-small sigset_t,
  // so we have to translate to struct sigaction64 first.
  struct sigaction64 kernel_new = {};
  if (bionic_new) {
    kernel_new.sa_flags = bionic_new->sa_flags;
    kernel_new.sa_handler = bionic_new->sa_handler;
#if defined(SA_RESTORER)
    kernel_new.sa_restorer = bionic_new->sa_restorer;
#endif
    // Don't filter signals here; if the caller asked for everything to be blocked, we should obey.
    memcpy(&kernel_new.sa_mask, &bionic_new->sa_mask, sizeof(bionic_new->sa_mask));
  }

  struct sigaction64 kernel_old;
  int result = __sigaction64(signal, bionic_new ? &kernel_new : nullptr, &kernel_old);
  if (bionic_old) {
    *bionic_old = {};
    bionic_old->sa_flags = kernel_old.sa_flags;
    bionic_old->sa_handler = kernel_old.sa_handler;
#if defined(SA_RESTORER)
    bionic_old->sa_restorer = kernel_old.sa_restorer;
#endif
    memcpy(&bionic_old->sa_mask, &kernel_old.sa_mask, sizeof(bionic_old->sa_mask));
  }
  return result;
}

int sigaction64(int signal, const struct sigaction64* bionic_new, struct sigaction64* bionic_old) {
  return __sigaction64(signal, bionic_new, bionic_old);
}

#endif
