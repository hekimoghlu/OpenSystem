/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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

#define UNIX64_RET_VOID		0
#define UNIX64_RET_UINT8	1
#define UNIX64_RET_UINT16	2
#define UNIX64_RET_UINT32	3
#define UNIX64_RET_SINT8	4
#define UNIX64_RET_SINT16	5
#define UNIX64_RET_SINT32	6
#define UNIX64_RET_INT64	7
#define UNIX64_RET_XMM32	8
#define UNIX64_RET_XMM64	9
#define UNIX64_RET_X87		10
#define UNIX64_RET_X87_2	11
#define UNIX64_RET_ST_XMM0_RAX	12
#define UNIX64_RET_ST_RAX_XMM0	13
#define UNIX64_RET_ST_XMM0_XMM1	14
#define UNIX64_RET_ST_RAX_RDX	15

#define UNIX64_RET_LAST		15

#define UNIX64_FLAG_RET_IN_MEM	(1 << 10)
#define UNIX64_FLAG_XMM_ARGS	(1 << 11)
#define UNIX64_SIZE_SHIFT	12

#if defined(FFI_EXEC_STATIC_TRAMP)
/*
 * For the trampoline code table mapping, a mapping size of 4K (base page size)
 * is chosen.
 */
#define UNIX64_TRAMP_MAP_SHIFT	12
#define UNIX64_TRAMP_MAP_SIZE	(1 << UNIX64_TRAMP_MAP_SHIFT)
#ifdef ENDBR_PRESENT
#define UNIX64_TRAMP_SIZE	40
#else
#define UNIX64_TRAMP_SIZE	32
#endif
#endif
