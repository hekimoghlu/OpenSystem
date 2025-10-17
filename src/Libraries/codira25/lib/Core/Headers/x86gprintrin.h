/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#ifndef __X86GPRINTRIN_H
#define __X86GPRINTRIN_H

#include <hresetintrin.h>

#include <uintrintrin.h>

#include <usermsrintrin.h>

#include <crc32intrin.h>

#include <prfchiintrin.h>

#include <raointintrin.h>

#include <cmpccxaddintrin.h>

#if defined(__i386__)
#define __SAVE_GPRBX "mov {%%ebx, %%eax |eax, ebx};"
#define __RESTORE_GPRBX "mov {%%eax, %%ebx |ebx, eax};"
#define __TMPGPR "eax"
#else
// When in 64-bit target, the 32-bit operands generate a 32-bit result,
// zero-extended to a 64-bit result in the destination general-purpose,
// It means "mov x %ebx" will clobber the higher 32 bits of rbx, so we
// should preserve the 64-bit register rbx.
#define __SAVE_GPRBX "mov {%%rbx, %%rax |rax, rbx};"
#define __RESTORE_GPRBX "mov {%%rax, %%rbx |rbx, rax};"
#define __TMPGPR "rax"
#endif

#define __SSC_MARK(__Tag)                                                      \
  __asm__ __volatile__( __SAVE_GPRBX                                           \
                       "mov {%0, %%ebx|ebx, %0}; "                             \
                       ".byte 0x64, 0x67, 0x90; "                              \
                        __RESTORE_GPRBX                                        \
                       ::"i"(__Tag)                                            \
                       :  __TMPGPR );

#endif /* __X86GPRINTRIN_H */
