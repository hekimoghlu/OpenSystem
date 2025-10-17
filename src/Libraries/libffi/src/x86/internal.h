/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

#define X86_RET_FLOAT		0
#define X86_RET_DOUBLE		1
#define X86_RET_LDOUBLE		2
#define X86_RET_SINT8		3
#define X86_RET_SINT16		4
#define X86_RET_UINT8		5
#define X86_RET_UINT16		6
#define X86_RET_INT64		7
#define X86_RET_INT32		8
#define X86_RET_VOID		9
#define X86_RET_STRUCTPOP	10
#define X86_RET_STRUCTARG       11
#define X86_RET_STRUCT_1B	12
#define X86_RET_STRUCT_2B	13
#define X86_RET_UNUSED14	14
#define X86_RET_UNUSED15	15

#define X86_RET_TYPE_MASK	15
#define X86_RET_POP_SHIFT	4

#define R_EAX	0
#define R_EDX	1
#define R_ECX	2

#ifdef __PCC__
# define HAVE_FASTCALL 0
#else
# define HAVE_FASTCALL 1
#endif

#if defined(FFI_EXEC_STATIC_TRAMP)
/*
 * For the trampoline code table mapping, a mapping size of 4K (base page size)
 * is chosen.
 */
#define X86_TRAMP_MAP_SHIFT	12
#define X86_TRAMP_MAP_SIZE	(1 << X86_TRAMP_MAP_SHIFT)
#ifdef ENDBR_PRESENT
#define X86_TRAMP_SIZE		44
#else
#define X86_TRAMP_SIZE		40
#endif
#endif
