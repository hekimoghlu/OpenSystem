/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#pragma once

/**
 * @file bits/strcasecmp.h
 * @brief Case-insensitive string comparison.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <xlocale.h>

__BEGIN_DECLS

/**
 * [strcasecmp(3)](https://man7.org/linux/man-pages/man3/strcasecmp.3.html) compares two strings
 * ignoring case.
 *
 * Returns an integer less than, equal to, or greater than zero if the first string is less than,
 * equal to, or greater than the second string (ignoring case).
 */
int strcasecmp(const char* _Nonnull __s1, const char* _Nonnull __s2) __attribute_pure__;

/**
 * Like strcasecmp() but taking a `locale_t`.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int strcasecmp_l(const char* _Nonnull __s1, const char* _Nonnull __s2, locale_t _Nonnull __l) __attribute_pure__ __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * [strncasecmp(3)](https://man7.org/linux/man-pages/man3/strncasecmp.3.html) compares the first
 * `n` bytes of two strings ignoring case.
 *
 * Returns an integer less than, equal to, or greater than zero if the first `n` bytes of the
 * first string is less than, equal to, or greater than the first `n` bytes of the second
 * string (ignoring case).
 */
int strncasecmp(const char* _Nonnull __s1, const char* _Nonnull __s2, size_t __n) __attribute_pure__;

/**
 * Like strncasecmp() but taking a `locale_t`.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int strncasecmp_l(const char* _Nonnull __s1, const char* _Nonnull __s2, size_t __n, locale_t _Nonnull __l) __attribute_pure__ __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
