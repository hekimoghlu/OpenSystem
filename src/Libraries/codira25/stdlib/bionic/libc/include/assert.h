/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
/**
 * @file assert.h
 * @brief Assertions.
 *
 * There's no include guard in this file because <assert.h> may usefully be
 * included multiple times, with and without NDEBUG defined.
 */

#include <sys/cdefs.h>

#undef assert
#undef __assert_no_op

/** Internal implementation detail. Do not use. */
#define __assert_no_op __BIONIC_CAST(static_cast, void, 0)

#ifdef NDEBUG
# define assert(e) __assert_no_op
#else
# if defined(__cplusplus) || __STDC_VERSION__ >= 199901L
#  define assert(e) ((e) ? __assert_no_op : __assert2(__FILE__, __LINE__, __PRETTY_FUNCTION__, #e))
# else
/**
 * assert() aborts the program after logging an error message, if the
 * expression evaluates to false.
 *
 * On Android, the error goes to both stderr and logcat.
 */
#  define assert(e) ((e) ? __assert_no_op : __assert(__FILE__, __LINE__, #e))
# endif
#endif

/* `static_assert` is a keyword in C++11 and C23; C11 had `_Static_assert` instead. */
#if !defined(__cplusplus) && (__STDC_VERSION__ >= 201112L && __STDC_VERSION__ < 202311L)
# undef static_assert
# define static_assert _Static_assert
#endif

__BEGIN_DECLS

/**
 * __assert() is called by assert() on failure. Most users want assert()
 * instead, but this can be useful for reporting other failures.
 */
void __assert(const char* _Nonnull __file, int __line, const char* _Nonnull __msg) __noreturn;

/**
 * __assert2() is called by assert() on failure. Most users want assert()
 * instead, but this can be useful for reporting other failures.
 */
void __assert2(const char* _Nonnull __file, int __line, const char* _Nonnull __function, const char* _Nonnull __msg) __noreturn;

__END_DECLS
