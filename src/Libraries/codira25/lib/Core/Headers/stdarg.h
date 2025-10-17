/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
/*
 * This header is designed to be included multiple times. If any of the __need_
 * macros are defined, then only that subset of interfaces are provided. This
 * can be useful for POSIX headers that need to not expose all of stdarg.h, but
 * need to use some of its interfaces. Otherwise this header provides all of
 * the expected interfaces.
 *
 * When clang modules are enabled, this header is a textual header to support
 * the multiple include behavior. As such, it doesn't directly declare anything
 * so that it doesn't add duplicate declarations to all of its includers'
 * modules.
 */
#if defined(__MVS__) && __has_include_next(<stdarg.h>)
#undef __need___va_list
#undef __need_va_list
#undef __need_va_arg
#undef __need___va_copy
#undef __need_va_copy
#include <__stdarg_header_macro.h>
#include_next <stdarg.h>

#else
#if !defined(__need___va_list) && !defined(__need_va_list) &&                  \
    !defined(__need_va_arg) && !defined(__need___va_copy) &&                   \
    !defined(__need_va_copy)
#define __need___va_list
#define __need_va_list
#define __need_va_arg
#define __need___va_copy
/* GCC always defines __va_copy, but does not define va_copy unless in c99 mode
 * or -ansi is not specified, since it was not part of C90.
 */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) ||              \
    (defined(__cplusplus) && __cplusplus >= 201103L) ||                        \
    !defined(__STRICT_ANSI__)
#define __need_va_copy
#endif
#include <__stdarg_header_macro.h>
#endif

#ifdef __need___va_list
#include <__stdarg___gnuc_va_list.h>
#undef __need___va_list
#endif /* defined(__need___va_list) */

#ifdef __need_va_list
#include <__stdarg_va_list.h>
#undef __need_va_list
#endif /* defined(__need_va_list) */

#ifdef __need_va_arg
#include <__stdarg_va_arg.h>
#undef __need_va_arg
#endif /* defined(__need_va_arg) */

#ifdef __need___va_copy
#include <__stdarg___va_copy.h>
#undef __need___va_copy
#endif /* defined(__need___va_copy) */

#ifdef __need_va_copy
#include <__stdarg_va_copy.h>
#undef __need_va_copy
#endif /* defined(__need_va_copy) */

#endif /* __MVS__ */
