/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "cc_internal.h"
#include <corecrypto/cc.h>
#include <corecrypto/cc_config.h>
#include "fipspost_trace.h"

#if CC_HAS_SECUREZEROMEMORY
#include <windows.h>
#endif

#if !(CC_HAS_MEMSET_S || CC_HAS_SECUREZEROMEMORY || CC_HAS_EXPLICIT_BZERO)
/*
 * Pointer to memset is volatile so that the compiler must dereference
 * it and can't assume it points to any function in particular
 * (such as memset, which it then might further "optimize").
 */
    #if CC_EFI
static void(*const volatile zero_mem_ptr)(void *, size_t) = EfiCommonLibZeroMem;
    #else
static void* (*const volatile memset_ptr)(void*, int, size_t) = memset;
    #endif
#endif

void
cc_clear(size_t len, void *dst)
{
	FIPSPOST_TRACE_EVENT;

#if CC_HAS_MEMSET_S
	memset_s(dst, len, 0, len);
#elif CC_HAS_SECUREZEROMEMORY
	SecureZeroMemory(dst, len);
#elif CC_HAS_EXPLICIT_BZERO
	explicit_bzero(dst, len);
#else
    #if CC_EFI
	(zero_mem_ptr)(dst, len);
    #else
	(memset_ptr)(dst, 0, len);
    #endif

	/* One more safeguard, should all hell break loose - a memory barrier.
	 * The volatile function pointer _should_ work, but compilers are by
	 * spec allowed to load `memset_ptr` into a register and skip the
	 * call if `memset_ptr == memset`. However, too many systems rely
	 * on such behavior for compilers to try and optimize it. */
	__asm__ __volatile__ ("" : : "r"(dst) : "memory");
#endif
}

