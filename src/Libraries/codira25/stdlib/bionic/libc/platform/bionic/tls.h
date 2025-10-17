/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#pragma once

#include <sys/cdefs.h>

#if defined(__aarch64__)

static inline void** __get_tls(void) {
  void** result;
  __asm__("mrs %0, tpidr_el0" : "=r"(result));
  return result;
}

static inline void __set_tls(void* tls) {
  __asm__("msr tpidr_el0, %0" : : "r" (tls));
}

#elif defined(__arm__)

static inline void** __get_tls(void) {
  void** result;
  __asm__("mrc p15, 0, %0, c13, c0, 3" : "=r"(result));
  return result;
}

// arm32 requires a syscall to set the thread pointer.
// By historical accident it's public API, but not in any header except this one.
__BEGIN_DECLS
int __set_tls(void* tls);
__END_DECLS

#elif defined(__i386__)

static inline void** __get_tls(void) {
  void** result;
  __asm__("movl %%gs:0, %0" : "=r"(result));
  return result;
}

// x86 is really hairy, so we keep that out of line.
__BEGIN_DECLS
int __set_tls(void* tls);
__END_DECLS

#elif defined(__riscv)

static inline void** __get_tls(void) {
  void** result;
  __asm__("mv %0, tp" : "=r"(result));
  return result;
}

static inline void __set_tls(void* tls) {
  __asm__("mv tp, %0" : : "r"(tls));
}

#elif defined(__x86_64__)

static inline void** __get_tls(void) {
  void** result;
  __asm__("mov %%fs:0, %0" : "=r"(result));
  return result;
}

// ARCH_SET_FS is not exposed via <sys/prctl.h> or <linux/prctl.h>.
#include <asm/prctl.h>
// This syscall stub is generated but it's not declared in any header.
__BEGIN_DECLS
int arch_prctl(int, unsigned long);
__END_DECLS

static inline int __set_tls(void* tls) {
  return arch_prctl(ARCH_SET_FS, reinterpret_cast<unsigned long>(tls));
}

#else
#error unsupported architecture
#endif

#include "tls_defines.h"
