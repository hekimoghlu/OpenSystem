/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
#ifndef _FD_SET
#define _FD_SET

#include <machine/types.h> /* __int32_t and uintptr_t */
#if !KERNEL
#include <Availability.h>
#endif

/*
 * Select uses bit masks of file descriptors in longs.  These macros
 * manipulate such bit fields (the filesystem macros use chars).  The
 * extra protection here is to permit application redefinition above
 * the default size.
 */
#ifdef FD_SETSIZE
#define __DARWIN_FD_SETSIZE     FD_SETSIZE
#else /* !FD_SETSIZE */
#define __DARWIN_FD_SETSIZE     1024
#endif /* FD_SETSIZE */
#define __DARWIN_NBBY           8                               /* bits in a byte */
#define __DARWIN_NFDBITS        (sizeof(__int32_t) * __DARWIN_NBBY) /* bits per mask */
#define __DARWIN_howmany(x, y)  ((((x) % (y)) == 0) ? ((x) / (y)) : (((x) / (y)) + 1)) /* # y's == x bits? */

__BEGIN_DECLS
typedef struct fd_set {
	__int32_t       fds_bits[__DARWIN_howmany(__DARWIN_FD_SETSIZE, __DARWIN_NFDBITS)];
} fd_set;

#if !KERNEL
int __darwin_check_fd_set_overflow(int, const void *, int) __API_AVAILABLE(macosx(10.16), ios(14.0), tvos(14.0), watchos(7.0));
#endif
__END_DECLS

#if !KERNEL
__header_always_inline int
__darwin_check_fd_set(int _a, const void *_b)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
#endif
	if ((uintptr_t)&__darwin_check_fd_set_overflow != (uintptr_t) 0) {
#if defined(_DARWIN_UNLIMITED_SELECT) || defined(_DARWIN_C_SOURCE)
		return __darwin_check_fd_set_overflow(_a, _b, 1);
#else
		return __darwin_check_fd_set_overflow(_a, _b, 0);
#endif
	} else {
		return 1;
	}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

/* This inline avoids argument side-effect issues with FD_ISSET() */
__header_always_inline int
__darwin_fd_isset(int _fd, const struct fd_set *_p)
{
	if (__darwin_check_fd_set(_fd, (const void *) _p)) {
		return _p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] & ((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS)));
	}

	return 0;
}

__header_always_inline void
__darwin_fd_set(int _fd, struct fd_set *const _p)
{
	if (__darwin_check_fd_set(_fd, (const void *) _p)) {
		(_p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] |= ((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS))));
	}
}

__header_always_inline void
__darwin_fd_clr(int _fd, struct fd_set *const _p)
{
	if (__darwin_check_fd_set(_fd, (const void *) _p)) {
		(_p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] &= ~((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS))));
	}
}

#else /* KERNEL */

__header_always_inline int
__darwin_fd_isset(int _fd, const struct fd_set *_p)
{
	return _p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] & ((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS)));
}

__header_always_inline void
__darwin_fd_set(int _fd, struct fd_set *const _p)
{
	(_p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] |= ((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS))));
}

__header_always_inline void
__darwin_fd_clr(int _fd, struct fd_set *const _p)
{
	(_p->fds_bits[(unsigned long)_fd / __DARWIN_NFDBITS] &= ~((__int32_t)(((unsigned long)1) << ((unsigned long)_fd % __DARWIN_NFDBITS))));
}
#endif /* KERNEL */

#define __DARWIN_FD_SET(n, p)   __darwin_fd_set((n), (p))
#define __DARWIN_FD_CLR(n, p)   __darwin_fd_clr((n), (p))
#define __DARWIN_FD_ISSET(n, p) __darwin_fd_isset((n), (p))

#if __GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 3
/*
 * Use the built-in bzero function instead of the library version so that
 * we do not pollute the namespace or introduce prototype warnings.
 */
#define __DARWIN_FD_ZERO(p)     __builtin_bzero(p, sizeof(*(p)))
#else
#define __DARWIN_FD_ZERO(p)     bzero(p, sizeof(*(p)))
#endif

#define __DARWIN_FD_COPY(f, t)  bcopy(f, t, sizeof(*(f)))
#endif /* _FD_SET */
