/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#ifndef _OS__OSBYTEORDERI386_H
#define _OS__OSBYTEORDERI386_H

#if defined(__i386__) || defined(__x86_64__)

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
	return (__uint16_t)((_data << 8) | (_data >> 8));
}

__DARWIN_OS_INLINE
__uint32_t
_OSSwapInt32(
	__uint32_t        _data
	)
{
#if defined(__llvm__)
	return __builtin_bswap32(_data);
#else
	__asm__ ("bswap   %0" : "+r" (_data));
	return _data;
#endif
}

#if defined(__llvm__)
__DARWIN_OS_INLINE
__uint64_t
_OSSwapInt64(
	__uint64_t        _data
	)
{
	return __builtin_bswap64(_data);
}

#elif defined(__i386__)
__DARWIN_OS_INLINE
__uint64_t
_OSSwapInt64(
	__uint64_t        _data
	)
{
	__asm__ ("bswap   %%eax\n\t"
                 "bswap   %%edx\n\t"
                 "xchgl   %%eax, %%edx"
                 : "+A" (_data));
	return _data;
}
#elif defined(__x86_64__)
__DARWIN_OS_INLINE
__uint64_t
_OSSwapInt64(
	__uint64_t        _data
	)
{
	__asm__ ("bswap   %0" : "+r" (_data));
	return _data;
}
#else
#error Unknown architecture
#endif

#endif /* defined(__i386__) || defined(__x86_64__) */

#endif /* ! _OS__OSBYTEORDERI386_H */
