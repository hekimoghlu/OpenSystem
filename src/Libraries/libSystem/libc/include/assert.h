/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#include <sys/cdefs.h>
#ifdef __cplusplus
#include <stdlib.h>
#endif /* __cplusplus */

/*
 * Unlike other ANSI header files, <assert.h> may usefully be included
 * multiple times, with and without NDEBUG defined.
 */

#undef assert
#undef __assert

#ifdef NDEBUG
#define	assert(e)	((void)0)
#else

#ifndef UNIFDEF_DRIVERKIT
#ifndef __GNUC__

__BEGIN_DECLS
#ifndef __cplusplus
void abort(void) __dead2 __cold;
#endif /* !__cplusplus */
int  printf(const char * __restrict, ...);
__END_DECLS

#define assert(e)  \
    ((void) ((e) ? ((void)0) : __assert (#e, __FILE__, __LINE__)))
#define __assert(e, file, line) \
    ((void)printf ("%s:%d: failed assertion `%s'\n", file, line, e), abort())

#else /* __GNUC__ */

__BEGIN_DECLS
void __assert_rtn(const char *, const char *, int, const char *) __dead2 __cold __disable_tail_calls;
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && ((__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0) < 1070)
void __eprintf(const char *, const char *, unsigned, const char *) __dead2 __cold;
#endif
__END_DECLS

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && ((__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0) < 1070)
#define __assert(e, file, line) \
    __eprintf ("%s:%d: failed assertion `%s'\n", file, line, e)
#else
/* 8462256: modified __assert_rtn() replaces deprecated __eprintf() */
#define __assert(e, file, line) \
    __assert_rtn ((const char *)-1L, file, line, e)
#endif

#if __DARWIN_UNIX03
#define	assert(e) \
    (__builtin_expect(!(e), 0) ? __assert_rtn(__func__, __FILE__, __LINE__, #e) : (void)0)
#else /* !__DARWIN_UNIX03 */
#define assert(e)  \
    (__builtin_expect(!(e), 0) ? __assert (#e, __FILE__, __LINE__) : (void)0)
#endif /* __DARWIN_UNIX03 */

#endif /* __GNUC__ */
#else /* UNIFDEF_DRIVERKIT */
__BEGIN_DECLS
void __assert_rtn(const char *, const char *, int, const char *) __dead2 __cold __disable_tail_calls;
__END_DECLS
#define	assert(e) \
    (__builtin_expect(!(e), 0) ? __assert_rtn(__func__, __FILE__, __LINE__, #e) : (void)0)
#endif /* UNIFDEF_DRIVERKIT */
#endif /* NDEBUG */

#ifndef _ASSERT_H_
#define _ASSERT_H_

#ifndef __cplusplus
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define static_assert _Static_assert
#endif /* __STDC_VERSION__ */
#endif /* !__cplusplus */

#endif /* _ASSERT_H_ */
