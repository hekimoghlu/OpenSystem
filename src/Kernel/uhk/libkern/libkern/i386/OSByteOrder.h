/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#ifndef _OS_OSBYTEORDERI386_H
#define _OS_OSBYTEORDERI386_H

#if defined(__i386__) || defined(__x86_64__)

#include <stdint.h>
#include <libkern/i386/_OSByteOrder.h>
#include <sys/_types/_os_inline.h>

/* Functions for byte reversed loads. */

OS_INLINE
uint16_t
OSReadSwapInt16(
	const volatile void   * base,
	uintptr_t       byteOffset
	)
{
	uint16_t result;

	result = *(volatile uint16_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt16(result);
}

OS_INLINE
uint32_t
OSReadSwapInt32(
	const volatile void   * base,
	uintptr_t       byteOffset
	)
{
	uint32_t result;

	result = *(volatile uint32_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt32(result);
}

OS_INLINE
uint64_t
OSReadSwapInt64(
	const volatile void   * base,
	uintptr_t       byteOffset
	)
{
	uint64_t result;

	result = *(volatile uint64_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt64(result);
}

/* Functions for byte reversed stores. */

OS_INLINE
void
OSWriteSwapInt16(
	volatile void   * base,
	uintptr_t       byteOffset,
	uint16_t        data
	)
{
	*(volatile uint16_t *)((uintptr_t)base + byteOffset) = _OSSwapInt16(data);
}

OS_INLINE
void
OSWriteSwapInt32(
	volatile void   * base,
	uintptr_t       byteOffset,
	uint32_t        data
	)
{
	*(volatile uint32_t *)((uintptr_t)base + byteOffset) = _OSSwapInt32(data);
}

OS_INLINE
void
OSWriteSwapInt64(
	volatile void    * base,
	uintptr_t        byteOffset,
	uint64_t         data
	)
{
	*(volatile uint64_t *)((uintptr_t)base + byteOffset) = _OSSwapInt64(data);
}

#endif /* defined(__i386__) || defined(__x86_64__) */

#endif /* ! _OS_OSBYTEORDERI386_H */
