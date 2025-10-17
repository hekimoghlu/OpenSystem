/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
 * @file err.h
 * @brief BSD error reporting functions. See `<error.h>` for the GNU equivalent.
 */

#include <sys/cdefs.h>

#include <stdarg.h>
#include <sys/types.h>

__BEGIN_DECLS

/**
 * [err(3)](https://man7.org/linux/man-pages/man3/err.3.html) outputs the program name,
 * the printf()-like formatted message, and the result of strerror() if `errno` is non-zero.
 *
 * Calls exit() with `__status`.
 *
 * New code should consider error() in `<error.h>`.
 */
__noreturn void err(int __status, const char* _Nullable __fmt, ...) __printflike(2, 3);

/**
 * [verr(3)](https://man7.org/linux/man-pages/man3/verr.3.html) outputs the program name,
 * the vprintf()-like formatted message, and the result of strerror() if `errno` is non-zero.
 *
 * Calls exit() with `__status`.
 *
 * New code should consider error() in `<error.h>`.
 */
__noreturn void verr(int __status, const char* _Nullable __fmt, va_list __args) __printflike(2, 0);

/**
 * [errx(3)](https://man7.org/linux/man-pages/man3/errx.3.html) outputs the program name, and
 * the printf()-like formatted message.
 *
 * Calls exit() with `__status`.
 *
 * New code should consider error() in `<error.h>`.
 */
__noreturn void errx(int __status, const char* _Nullable __fmt, ...) __printflike(2, 3);

/**
 * [verrx(3)](https://man7.org/linux/man-pages/man3/verrx.3.html) outputs the program name, and
 * the vprintf()-like formatted message.
 *
 * Calls exit() with `__status`.
 *
 * New code should consider error() in `<error.h>`.
 */
__noreturn void verrx(int __status, const char* _Nullable __fmt, va_list __args) __printflike(2, 0);

/**
 * [warn(3)](https://man7.org/linux/man-pages/man3/warn.3.html) outputs the program name,
 * the printf()-like formatted message, and the result of strerror() if `errno` is non-zero.
 *
 * New code should consider error() in `<error.h>`.
 */
void warn(const char* _Nullable __fmt, ...) __printflike(1, 2);

/**
 * [vwarn(3)](https://man7.org/linux/man-pages/man3/vwarn.3.html) outputs the program name,
 * the vprintf()-like formatted message, and the result of strerror() if `errno` is non-zero.
 *
 * New code should consider error() in `<error.h>`.
 */
void vwarn(const char* _Nullable __fmt, va_list __args) __printflike(1, 0);

/**
 * [warnx(3)](https://man7.org/linux/man-pages/man3/warnx.3.html) outputs the program name, and
 * the printf()-like formatted message.
 *
 * New code should consider error() in `<error.h>`.
 */
void warnx(const char* _Nullable __fmt, ...) __printflike(1, 2);

/**
 * [vwarnx(3)](https://man7.org/linux/man-pages/man3/vwarnx.3.html) outputs the program name, and
 * the vprintf()-like formatted message.
 *
 * New code should consider error() in `<error.h>`.
 */
void vwarnx(const char* _Nullable __fmt, va_list __args) __printflike(1, 0);

__END_DECLS
