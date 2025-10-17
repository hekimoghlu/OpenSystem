/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
/*
 *	Copyright (c) 1998, Apple Computer Inc. All rights reserved.
 *
 *	File: _setjmp.h
 *
 *	Defines for register offsets in the save area.
 *
 */

#if defined(__arm__)

#define JMP_r4		0x00
#define JMP_r5		0x04
#define JMP_r6		0x08
#define JMP_r7		0x0c
#define JMP_r8		0x10
#define JMP_r10		0x14
#define JMP_fp		0x18
#define JMP_sp		0x1c
#define JMP_lr		0x20

#define JMP_VFP		0x24

#define JMP_sigmask	0x68
#define JMP_sigonstack	0x6C

#define STACK_SSFLAGS	8 // offsetof(stack_t, ss_flags)


#define JMP_SIGFLAG	0x70

#else
#error architecture not supported
#endif
