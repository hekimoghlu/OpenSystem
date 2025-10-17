/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

// An inline version of pthread_sigqueue(pthread_self(), ...), to reduce the number of
// uninteresting stack frames at the top of a crash.
static inline __always_inline void inline_raise(int sig, void* value = nullptr) {
  // Protect ourselves against stale cached PID/TID values by fetching them via syscall.
  // http://b/37769298
  pid_t pid = syscall(__NR_getpid);
  pid_t tid = syscall(__NR_gettid);
  siginfo_t info = {};
  info.si_code = SI_QUEUE;
  info.si_pid = pid;
  info.si_uid = getuid();
  info.si_value.sival_ptr = value;

#if defined(__arm__)
  register long r0 __asm__("r0") = pid;
  register long r1 __asm__("r1") = tid;
  register long r2 __asm__("r2") = sig;
  register long r3 __asm__("r3") = reinterpret_cast<long>(&info);
  register long r7 __asm__("r7") = __NR_rt_tgsigqueueinfo;
  __asm__("swi #0" : "=r"(r0) : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r7) : "memory");
#elif defined(__aarch64__)
  register long x0 __asm__("x0") = pid;
  register long x1 __asm__("x1") = tid;
  register long x2 __asm__("x2") = sig;
  register long x3 __asm__("x3") = reinterpret_cast<long>(&info);
  register long x8 __asm__("x8") = __NR_rt_tgsigqueueinfo;
  __asm__("svc #0" : "=r"(x0) : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(x8) : "memory");
#elif defined(__riscv)
  register long a0 __asm__("a0") = pid;
  register long a1 __asm__("a1") = tid;
  register long a2 __asm__("a2") = sig;
  register long a3 __asm__("a3") = reinterpret_cast<long>(&info);
  register long a7 __asm__("a7") = __NR_rt_tgsigqueueinfo;
  __asm__("ecall" : "=r"(a0) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a7) : "memory");
#elif defined(__x86_64__)
  register long rax __asm__("rax") = __NR_rt_tgsigqueueinfo;
  register long rdi __asm__("rdi") = pid;
  register long rsi __asm__("rsi") = tid;
  register long rdx __asm__("rdx") = sig;
  register long r10 __asm__("r10") = reinterpret_cast<long>(&info);
  __asm__("syscall"
          : "+r"(rax)
          : "r"(rdi), "r"(rsi), "r"(rdx), "r"(r10)
          : "memory", "cc", "r11", "rcx");
#else
  // 32-bit x86 is a huge mess, so don't even bother...
  syscall(__NR_rt_tgsigqueueinfo, pid, tid, sig, &info);
#endif
}
