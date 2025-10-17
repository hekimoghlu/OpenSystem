/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
#ifndef _SAN_MEMINTRINSICS_H_
#define _SAN_MEMINTRINSICS_H_

#include <sys/cdefs.h>

/*
 * Non-sanitized versions of memory intrinsics
 */
static inline void *
__sized_by(sz)
__nosan_memcpy(void *dst __sized_by(sz), const void *src __sized_by(sz), size_t sz)
{
	return memcpy(dst, src, sz);
}
static inline void *
__sized_by(sz)
__nosan_memset(void *dst __sized_by(sz), int c, size_t sz)
{
	return memset(dst, c, sz);
}
static inline void *
__sized_by(sz)
__nosan_memmove(void *dst __sized_by(sz), const void *src __sized_by(sz), size_t sz)
{
	return memmove(dst, src, sz);
}
static inline int
__nosan_bcmp(const void *a __sized_by(sz), const void *b __sized_by(sz), size_t sz)
{
	return bcmp(a, b, sz);
}
static inline void
__nosan_bcopy(const void *src __sized_by(sz), void *dst __sized_by(sz), size_t sz)
{
	bcopy(src, dst, sz);
}
static inline int
__nosan_memcmp(const void *a __sized_by(sz), const void *b __sized_by(sz), size_t sz)
{
	return memcmp(a, b, sz);
}
static inline void
__nosan_bzero(void *dst __sized_by(sz), size_t sz)
{
	bzero(dst, sz);
}

static inline size_t
__nosan_strlcpy(char *__sized_by(sz)dst, const char *__null_terminated src, size_t sz)
{
	return strlcpy(dst, src, sz);
}
static inline size_t
__nosan_strlcat(char *__sized_by(sz)dst, const char *__null_terminated src, size_t sz)
{
	return strlcat(dst, src, sz);
}
static inline size_t
__nosan_strnlen(const char *__counted_by(sz)src, size_t sz)
{
	return strnlen(src, sz);
}
static inline size_t
__nosan_strlen(const char *__null_terminated src)
{
	return strlen(src);
}
#if !__has_ptrcheck && !__has_include(<__xnu_libcxx_sentinel.h>)
static inline char *
__nosan_strncpy(char *dst, const char *src, size_t sz)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
	return strncpy(dst, src, sz);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}
static inline char *
__nosan_strncat(char *dst, const char *src, size_t sz)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
	return strncat(dst, src, sz);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}
#endif /* !__has_ptrcheck && !__has_include(<__xnu_libcxx_sentinel.h>) */

#if KASAN
void *__sized_by(sz) __asan_memcpy(void *dst __sized_by(sz), const void *src __sized_by(sz), size_t sz);
void *__sized_by(sz) __asan_memset(void * __sized_by(sz), int c, size_t sz);
void *__sized_by(sz) __asan_memmove(void *dst __sized_by(sz), const void *src __sized_by(sz), size_t sz);
void  __asan_bcopy(const void *src __sized_by(sz), void *dst __sized_by(sz), size_t sz);
void  __asan_bzero(void *dst __sized_by(sz), size_t sz);
int   __asan_bcmp(const void *a __sized_by(sz), const void *b __sized_by(sz), size_t sz);
int   __asan_memcmp(const void *a __sized_by(sz), const void *b __sized_by(sz), size_t sz);

size_t __asan_strlcpy(char *__sized_by(sz) dst, const char *__null_terminated src, size_t sz);
char  *__asan_strncpy(char *dst, const char *src, size_t sz);
char  *__asan_strncat(char *dst, const char *src, size_t sz);
size_t __asan_strlcat(char *__sized_by(sz) dst, const char *__null_terminated src, size_t sz);
size_t __asan_strnlen(const char *__null_terminated src, size_t sz);
size_t __asan_strlen(const char *__null_terminated src);

#define memcpy    __asan_memcpy
#define memmove   __asan_memmove
#define memset    __asan_memset
#define bcopy     __asan_bcopy
#define bzero     __asan_bzero
#define bcmp      __asan_bcmp
#define memcmp    __asan_memcmp

#define strlcpy   __asan_strlcpy
#define strncpy   __asan_strncpy
#define strlcat   __asan_strlcat
#define strncat   __asan_strncat
// #define strnlen   __asan_strnlen
// #define strlen    __asan_strlen

#endif

#endif /* _SAN_MEMINTRINSICS_H_ */
