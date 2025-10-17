/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#ifndef _OS__OSBYTEORDER_H
#define _OS__OSBYTEORDER_H

#include <sys/_types.h>

/* Macros for swapping constant values in the preprocessing stage. */
#define __DARWIN_OSSwapConstInt16(x) \
    ((__uint16_t)((((__uint16_t)(x) & 0xff00U) >> 8) | \
	        (((__uint16_t)(x) & 0x00ffU) << 8)))

#define __DARWIN_OSSwapConstInt32(x) \
    ((__uint32_t)((((__uint32_t)(x) & 0xff000000U) >> 24) | \
	        (((__uint32_t)(x) & 0x00ff0000U) >>  8) | \
	        (((__uint32_t)(x) & 0x0000ff00U) <<  8) | \
	        (((__uint32_t)(x) & 0x000000ffU) << 24)))

#define __DARWIN_OSSwapConstInt64(x) \
    ((__uint64_t)((((__uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
	        (((__uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
	        (((__uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
	        (((__uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
	        (((__uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
	        (((__uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
	        (((__uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
	        (((__uint64_t)(x) & 0x00000000000000ffULL) << 56)))

#if defined(__GNUC__)

#if defined(__i386__) || defined(__x86_64__)
#include <libkern/i386/_OSByteOrder.h>
#endif

#if defined (__arm__) || defined(__arm64__)
#include <libkern/arm/_OSByteOrder.h>
#endif


#define __DARWIN_OSSwapInt16(x) \
    ((__uint16_t)(__builtin_constant_p(x) ? __DARWIN_OSSwapConstInt16(x) : _OSSwapInt16(x)))

#define __DARWIN_OSSwapInt32(x) \
    (__builtin_constant_p(x) ? __DARWIN_OSSwapConstInt32(x) : _OSSwapInt32(x))

#define __DARWIN_OSSwapInt64(x) \
    (__builtin_constant_p(x) ? __DARWIN_OSSwapConstInt64(x) : _OSSwapInt64(x))

#else /* ! __GNUC__ */

#if defined(__i386__) || defined(__x86_64__)

#if !defined(__DARWIN_OS_INLINE)
# if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#        define __DARWIN_OS_INLINE static inline
# elif defined(__MWERKS__) || defined(__cplusplus)
#        define __DARWIN_OS_INLINE static inline
# else
#        define __DARWIN_OS_INLINE static __inline__
# endif
#endif

__DARWIN_OS_INLINE
__uint16_t
_OSSwapInt16(
	__uint16_t                    data
	)
{
	return __DARWIN_OSSwapConstInt16(data);
}

__DARWIN_OS_INLINE
__uint32_t
_OSSwapInt32(
	__uint32_t                    data
	)
{
	return __DARWIN_OSSwapConstInt32(data);
}

__DARWIN_OS_INLINE
__uint64_t
_OSSwapInt64(
	__uint64_t                    data
	)
{
	return __DARWIN_OSSwapConstInt64(data);
}
#endif

#define __DARWIN_OSSwapInt16(x) _OSSwapInt16(x)

#define __DARWIN_OSSwapInt32(x) _OSSwapInt32(x)

#define __DARWIN_OSSwapInt64(x) _OSSwapInt64(x)

#endif /* __GNUC__ */

#endif /* ! _OS__OSBYTEORDER_H */
