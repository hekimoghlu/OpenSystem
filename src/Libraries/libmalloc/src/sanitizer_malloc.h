/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#ifndef _SANITIZER_MALLOC_H_
#define _SANITIZER_MALLOC_H_

#include "base.h"
#include "malloc/malloc.h"
#include <stdbool.h>

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

MALLOC_NOEXPORT
bool
sanitizer_should_enable(void);

MALLOC_NOEXPORT
void
sanitizer_reset_environment(void);

MALLOC_NOEXPORT
malloc_zone_t *
sanitizer_create_zone(malloc_zone_t *wrapped_zone);

static inline uint16_t
_malloc_read_uint16_via_rsp(const void *ptr)
{
#if TARGET_CPU_X86_64
	__asm__ (
		"subq  %%rsp,       %0  \n"
		"movw (%%rsp, %0),  %w0 \n"
		: "+r" (ptr)            // outputs, ptr = %0 read-write
		:                       // inputs, empty
		:                       // clobbers, empty
	);
	return (uint16_t)(uintptr_t)ptr;
#elif TARGET_CPU_ARM64 && TARGET_RT_64_BIT
	__asm__ (
		"sub  %0, %0, fp    \n"
		"ldrh %w0, [fp, %0] \n"
		: "+r" (ptr)            // outputs, ptr = %0 read-write
		:                       // inputs, empty
		:                       // clobbers, empty
	);
	return (uint16_t)(uintptr_t)ptr;
#else
	return *(uint16_t *)ptr;
#endif
}

static inline uint64_t
_malloc_read_uint64_via_rsp(const void *ptr)
{
#if TARGET_CPU_X86_64
	__asm__ (
		"subq  %%rsp,       %0  \n"
		"movq (%%rsp, %0),  %0  \n"
		: "+r" (ptr)            // outputs, ptr = %0 read-write
		:                       // inputs, empty
		:                       // clobbers, empty
	);
	return (uint64_t)ptr;
#elif TARGET_CPU_ARM64 && TARGET_RT_64_BIT
	__asm__ (
		"sub %0, %0, fp  \n"
		"ldr %0, [fp, %0]\n"
		: "+r" (ptr)            // outputs, ptr = %0 read-write
		:                       // inputs, empty
		:                       // clobbers, empty
	);
	return (uint64_t)ptr;
#else
	return *(uint64_t *)ptr;
#endif
}

static inline void
_malloc_write_uint64_via_rsp(void *ptr, uint64_t value)
{
#if TARGET_CPU_X86_64
	__asm__ (
		"subq  %%rsp,  %0         \n"
		"movq  %1,    (%%rsp, %0) \n"
		:                         // outputs, empty
		: "r" (ptr), "r" (value)  // inputs, ptr = %0, value = %1
		:                         // clobbers, empty
	);
#elif TARGET_CPU_ARM64 && TARGET_RT_64_BIT
	__asm__ volatile (
		"sub %0, %0, fp   \n"
		"str %1, [fp, %0] \n"
		: "+r" (ptr)              // outputs, ptr = %0 (not a real output but gets clobbered)
		: "r" (value)             // inputs, value = %1
		:                         // clobbers, empty
	);
#else
	*(uint64_t *)ptr = value;
#endif
}

#endif // _SANITIZER_MALLOC_H_
