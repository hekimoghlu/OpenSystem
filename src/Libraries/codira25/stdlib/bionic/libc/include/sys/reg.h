/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#ifndef _SYS_REG_H_
#define _SYS_REG_H_

#include <sys/cdefs.h>

#if defined(__i386__)

#define EBX 0
#define ECX 1
#define EDX 2
#define ESI 3
#define EDI 4
#define EBP 5
#define EAX 6
#define DS 7
#define ES 8
#define FS 9
#define GS 10
#define ORIG_EAX 11
#define EIP 12
#define CS 13
#define EFL 14
#define UESP 15
#define SS 16

#elif defined(__x86_64__)

#define R15 0
#define R14 1
#define R13 2
#define R12 3
#define RBP 4
#define RBX 5
#define R11 6
#define R10 7
#define R9 8
#define R8 9
#define RAX 10
#define RCX 11
#define RDX 12
#define RSI 13
#define RDI 14
#define ORIG_RAX 15
#define RIP 16
#define CS 17
#define EFLAGS 18
#define RSP 19
#define SS 20
#define FS_BASE 21
#define GS_BASE 22
#define DS 23
#define ES 24
#define FS 25
#define GS 26

#endif

#endif
