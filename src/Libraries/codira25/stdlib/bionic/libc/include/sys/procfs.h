/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include <sys/ptrace.h>
#include <sys/ucontext.h>

__BEGIN_DECLS

#if defined(__arm__)
#define ELF_NGREG (sizeof(struct user_regs) / sizeof(elf_greg_t))
#elif defined(__aarch64__)
#define ELF_NGREG (sizeof(struct user_pt_regs) / sizeof(elf_greg_t))
#else
#define ELF_NGREG (sizeof(struct user_regs_struct) / sizeof(elf_greg_t))
#endif

typedef unsigned long elf_greg_t;
typedef elf_greg_t elf_gregset_t[ELF_NGREG];

typedef fpregset_t elf_fpregset_t;

#if defined(__i386__)
typedef struct user_fpxregs_struct elf_fpxregset_t;
#endif

typedef elf_gregset_t prgregset_t;
typedef elf_fpregset_t prfpregset_t;

typedef pid_t lwpid_t;
typedef void* psaddr_t;

struct elf_siginfo {
  int si_signo;
  int si_code;
  int si_errno;
};

#define ELF_PRARGSZ 80

__END_DECLS
