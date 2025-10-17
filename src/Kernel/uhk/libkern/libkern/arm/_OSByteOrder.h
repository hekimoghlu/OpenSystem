/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
#ifndef _OS__OSBYTEORDERARM_H
#define _OS__OSBYTEORDERARM_H

#if defined (__arm__) || defined(__arm64__)

#include <sys/_types.h>

#if !defined(__DARWIN_OS_INLINE)
# if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#        define __DARWIN_OS_INLINE static inline
# elif defined(__MWERKS__) || defined(__cplusplus)
#        define __DARWIN_OS_INLINE static inline
# else
#        define __DARWIN_OS_INLINE static __inline__
# endif
#endif

/* Generic byte swapping functions. */

__DARWIN_OS_INLINE
__uint16_t
_OSSwapInt16(
	__uint16_t        _data
	)
{
	/* Reduces to 'rev16' with clang */
	return (__uint16_t)(_data << 8 | _data >> 8);
}

__DARWIN_OS_INLINE
__uint32_t
_OSSwapInt32(
	__uint32_t        _data
	)
{
#if defined(__llvm__)
	_data = __builtin_bswap32(_data);
#else
	/* This actually generates the best code */
	_data = (((_data ^ (_data >> 16 | (_data << 16))) & 0xFF00FFFF) >> 8) ^ (_data >> 8 | _data << 24);
#endif

	return _data;
}

__DARWIN_OS_INLINE
__uint64_t
_OSSwapInt64(
	__uint64_t        _data
	)
{
#if defined(__llvm__)
	return __builtin_bswap64(_data);
#else
	union {
		__uint64_t _ull;
		__uint32_t _ul[2];
	} _u;

	/* This actually generates the best code */
	_u._ul[0] = (__uint32_t)(_data >> 32);
	_u._ul[1] = (__uint32_t)(_data & 0xffffffff);
	_u._ul[0] = _OSSwapInt32(_u._ul[0]);
	_u._ul[1] = _OSSwapInt32(_u._ul[1]);
	return _u._ull;
#endif
}

#endif /* defined (__arm__) || defined(__arm64__) */

#endif /* ! _OS__OSBYTEORDERARM_H */
