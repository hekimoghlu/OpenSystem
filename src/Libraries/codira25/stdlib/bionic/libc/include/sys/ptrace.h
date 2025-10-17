/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#ifndef _SYS_PTRACE_H_
#define _SYS_PTRACE_H_

#include <sys/cdefs.h>
#include <sys/types.h>
#include <linux/ptrace.h>

__BEGIN_DECLS

/* glibc uses different PTRACE_ names from the kernel for these two... */
#define PTRACE_POKEUSER PTRACE_POKEUSR
#define PTRACE_PEEKUSER PTRACE_PEEKUSR

/* glibc exports a different set of PT_ names too... */
#define PT_TRACE_ME PTRACE_TRACEME
#define PT_READ_I PTRACE_PEEKTEXT
#define PT_READ_D PTRACE_PEEKDATA
#define PT_READ_U PTRACE_PEEKUSR
#define PT_WRITE_I PTRACE_POKETEXT
#define PT_WRITE_D PTRACE_POKEDATA
#define PT_WRITE_U PTRACE_POKEUSR
#define PT_CONT PTRACE_CONT
#define PT_KILL PTRACE_KILL
#define PT_STEP PTRACE_SINGLESTEP
#define PT_GETFPREGS PTRACE_GETFPREGS
#define PT_ATTACH PTRACE_ATTACH
#define PT_DETACH PTRACE_DETACH
#define PT_SYSCALL PTRACE_SYSCALL
#define PT_SETOPTIONS PTRACE_SETOPTIONS
#define PT_GETEVENTMSG PTRACE_GETEVENTMSG
#define PT_GETSIGINFO PTRACE_GETSIGINFO
#define PT_SETSIGINFO PTRACE_SETSIGINFO

long ptrace(int __op, ...);

__END_DECLS

#endif
