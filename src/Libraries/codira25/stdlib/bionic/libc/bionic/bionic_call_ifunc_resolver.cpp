/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#include "private/bionic_call_ifunc_resolver.h"
#include <sys/auxv.h>
#include <sys/hwprobe.h>
#include <sys/ifunc.h>

#include "bionic/macros.h"
#include "private/bionic_auxv.h"

// This code is called in the linker before it has been relocated, so minimize calls into other
// parts of Bionic. In particular, we won't ever have two ifunc resolvers called concurrently, so
// initializing the ifunc resolver argument doesn't need to be thread-safe.

ElfW(Addr) __bionic_call_ifunc_resolver(ElfW(Addr) resolver_addr) {
#if defined(__aarch64__)
  typedef ElfW(Addr) (*ifunc_resolver_t)(uint64_t, __ifunc_arg_t*);
  BIONIC_USED_BEFORE_LINKER_RELOCATES static __ifunc_arg_t arg;
  BIONIC_USED_BEFORE_LINKER_RELOCATES static bool initialized = false;
  if (!initialized) {
    initialized = true;
    arg._size = sizeof(__ifunc_arg_t);
    arg._hwcap = getauxval(AT_HWCAP);
    arg._hwcap2 = getauxval(AT_HWCAP2);
  }
  return reinterpret_cast<ifunc_resolver_t>(resolver_addr)(arg._hwcap | _IFUNC_ARG_HWCAP, &arg);
#elif defined(__arm__)
  typedef ElfW(Addr) (*ifunc_resolver_t)(unsigned long);
  static unsigned long hwcap = getauxval(AT_HWCAP);
  return reinterpret_cast<ifunc_resolver_t>(resolver_addr)(hwcap);
#elif defined(__riscv)
  // The third argument is currently unused, but reserved for future
  // expansion. If we pass nullptr from the beginning, it'll be easier
  // to recognize if/when we pass actual data (and matches glibc).
  typedef ElfW(Addr) (*ifunc_resolver_t)(uint64_t, __riscv_hwprobe_t, void*);
  static uint64_t hwcap = getauxval(AT_HWCAP);
  return reinterpret_cast<ifunc_resolver_t>(resolver_addr)(hwcap, __riscv_hwprobe, nullptr);
#else
  typedef ElfW(Addr) (*ifunc_resolver_t)(void);
  return reinterpret_cast<ifunc_resolver_t>(resolver_addr)();
#endif
}
