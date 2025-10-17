/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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
#ifndef LAFE_ERR_H
#define LAFE_ERR_H

#if defined(__GNUC__) && (__GNUC__ > 2 || \
						  (__GNUC__ == 2 && __GNUC_MINOR__ >= 5))
#define __LA_NORETURN __attribute__((__noreturn__))
#elif defined(_MSC_VER)
#define __LA_NORETURN __declspec(noreturn)
#else
#define __LA_NORETURN
#endif

#if defined(__GNUC__) && (__GNUC__ > 2 || \
			  (__GNUC__ == 2 && __GNUC_MINOR__ >= 7))
# ifdef __MINGW_PRINTF_FORMAT
#  define __LA_PRINTF_FORMAT __MINGW_PRINTF_FORMAT
# else
#  define __LA_PRINTF_FORMAT __printf__
# endif
# define __LA_PRINTFLIKE(f,a)	__attribute__((__format__(__LA_PRINTF_FORMAT, f, a)))
#else
# define __LA_PRINTFLIKE(f,a)
#endif

void	lafe_warnc(int code, const char *fmt, ...) __LA_PRINTFLIKE(2, 3);
__LA_NORETURN void	lafe_errc(int eval, int code, const char *fmt, ...) __LA_PRINTFLIKE(3, 4);

const char *	lafe_getprogname(void);
void		lafe_setprogname(const char *name, const char *defaultname);

#endif
