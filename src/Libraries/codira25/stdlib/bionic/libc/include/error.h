/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
 * @file error.h
 * @brief GNU error reporting functions.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * [error_print_progname(3)](https://man7.org/linux/man-pages/man3/error_print_progname.3.html) is
 * a function pointer that, if non-null, is called by error() instead of prefixing errors with the
 * program name.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
extern void (* _Nullable error_print_progname)(void) __INTRODUCED_IN(23);

/**
 * [error_message_count(3)](https://man7.org/linux/man-pages/man3/error_message_count.3.html) is
 * a global count of the number of calls to error() and error_at_line().
 *
 * Available since API level 23.
 */
extern unsigned int error_message_count __INTRODUCED_IN(23);

/**
 * [error_one_per_line(3)](https://man7.org/linux/man-pages/man3/error_one_per_line.3.html) is
 * a global flag that if non-zero disables printing multiple errors with the same filename and
 * line number.
 *
 * Available since API level 23.
 */
extern int error_one_per_line __INTRODUCED_IN(23);

/**
 * [error(3)](https://man7.org/linux/man-pages/man3/error.3.html) formats the given printf()-like
 * error message, preceded by the program name. Calls exit if `__status` is non-zero, and appends
 * the result of strerror() if `__errno` is non-zero.
 *
 * Available since API level 23.
 */
void error(int __status, int __errno, const char* _Nonnull __fmt, ...) __printflike(3, 4) __INTRODUCED_IN(23);

/**
 * [error_at_line(3)](https://man7.org/linux/man-pages/man3/error_at_line.3.html) formats the given
 * printf()-like error message, preceded by the program name and the given filename and line number.
 * Calls exit if `__status` is non-zero, and appends the result of strerror() if `__errno` is
 * non-zero.
 *
 * Available since API level 23.
 */
void error_at_line(int __status, int __errno, const char* _Nonnull __filename, unsigned int __line_number, const char* _Nonnull __fmt, ...) __printflike(5, 6) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
