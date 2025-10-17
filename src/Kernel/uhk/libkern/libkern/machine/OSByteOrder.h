/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#ifndef _OS_OSBYTEORDERMACHINE_H
#define _OS_OSBYTEORDERMACHINE_H

#if !defined(__GNUC__) || (!defined(__i386__) && !defined(__x86_64__) && !defined (__arm__) && !defined(__arm64__))

#include <stdint.h>

#if !defined(OS_INLINE)
# if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#        define OS_INLINE static inline
# elif defined(__MWERKS__) || defined(__cplusplus)
#        define OS_INLINE static inline
# else
#        define OS_INLINE static __inline__
# endif
#endif

/* Generic byte swapping functions. */

OS_INLINE
uint16_t
_OSSwapInt16(
	uint16_t                      data
	)
{
	return OSSwapConstInt16(data);
}

OS_INLINE
uint32_t
_OSSwapInt32(
	uint32_t                      data
	)
{
	return OSSwapConstInt32(data);
}

OS_INLINE
uint64_t
_OSSwapInt64(
	uint64_t                        data
	)
{
	return OSSwapConstInt64(data);
}

/* Functions for byte reversed loads. */

OS_INLINE
uint16_t
OSReadSwapInt16(
	const volatile void               * base,
	uintptr_t                     byteOffset
	)
{
	uint16_t data = *(volatile uint16_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt16(data);
}

OS_INLINE
uint32_t
OSReadSwapInt32(
	const volatile void               * base,
	uintptr_t                     byteOffset
	)
{
	uint32_t data = *(volatile uint32_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt32(data);
}

OS_INLINE
uint64_t
OSReadSwapInt64(
	const volatile void               * base,
	uintptr_t                     byteOffset
	)
{
	uint64_t data = *(volatile uint64_t *)((uintptr_t)base + byteOffset);
	return _OSSwapInt64(data);
}

/* Functions for byte reversed stores. */

OS_INLINE
void
OSWriteSwapInt16(
	volatile void               * base,
	uintptr_t                     byteOffset,
	uint16_t                      data
	)
{
	*(volatile uint16_t *)((uintptr_t)base + byteOffset) = _OSSwapInt16(data);
}

OS_INLINE
void
OSWriteSwapInt32(
	volatile void               * base,
	uintptr_t                     byteOffset,
	uint32_t                      data
	)
{
	*(volatile uint32_t *)((uintptr_t)base + byteOffset) = _OSSwapInt32(data);
}

OS_INLINE
void
OSWriteSwapInt64(
	volatile void               * base,
	uintptr_t                     byteOffset,
	uint64_t                      data
	)
{
	*(volatile uint64_t *)((uintptr_t)base + byteOffset) = _OSSwapInt64(data);
}

#endif /* !defined(__GNUC__) || (!defined(__i386__) && !defined(__x86_64__) && !defined (__arm__) && !defined(__arm64__)) */

#endif /* ! _OS_OSBYTEORDERMACHINE_H */
