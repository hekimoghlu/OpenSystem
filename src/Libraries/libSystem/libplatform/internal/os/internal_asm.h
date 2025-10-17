/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#ifndef __INTERNAL_ASM_H__
#define __INTERNAL_ASM_H__

#include <machine/cpu_capabilities.h>

#define OS_STRINGIFY1(s) #s
#define OS_STRINGIFY(s) OS_STRINGIFY1(s)
#define OS_CONCAT1(x, y) x ## y
#define OS_CONCAT(x, y) OS_CONCAT1(x, y)

#ifdef	__ASSEMBLER__
#define OS_VARIANT(f, v) OS_CONCAT(_, OS_CONCAT(f, OS_CONCAT($VARIANT$, v)))
#else
#define OS_VARIANT(f, v) OS_CONCAT(f, OS_CONCAT($VARIANT$, v))
#endif

#if defined(__ASSEMBLER__)

#if defined(__i386__) || defined(__x86_64__)

#define OS_VARIANT_FUNCTION_START(name, variant, alignment) \
	.text ; \
	.align alignment, 0x90 ; \
	.private_extern OS_VARIANT(name, variant) ; \
	OS_VARIANT(name, variant) ## :

// GENERIC indicates that this function will be chosen as the generic
// implementation (at compile time) when building targets which do not
// support dyld variant resolves.
#if defined(VARIANT_NO_RESOLVERS) || defined(VARIANT_DYLD)
#define OS_VARIANT_FUNCTION_START_GENERIC(name, variant, alignment) \
	OS_VARIANT_FUNCTION_START(name, variant, alignment) \
	.globl _ ## name ; \
	_ ## name ## :
#else
#define OS_VARIANT_FUNCTION_START_GENERIC OS_VARIANT_FUNCTION_START
#endif

#define OS_ATOMIC_FUNCTION_START(name, alignment) \
	.text ; \
	.align alignment, 0x90 ; \
	.globl _ ## name ; \
	_ ## name ## :

#endif // defined(__i386__) || defined(__x86_64__)

#endif // defined(__ASSEMBLER__)

#endif // __INTERNAL_ASM_H__
