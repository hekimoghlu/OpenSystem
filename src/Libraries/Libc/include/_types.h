/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#ifndef __TYPES_H_
#define __TYPES_H_

#include <sys/_types.h>
#include <_bounds.h>
#include <machine/_types.h> /* __uint32_t */

_LIBC_SINGLE_BY_DEFAULT()

#if __GNUC__ > 2 || __GNUC__ == 2 && __GNUC_MINOR__ >= 7
#define __strfmonlike(fmtarg, firstvararg) \
		__attribute__((__format__ (__strfmon__, fmtarg, firstvararg)))
#define __strftimelike(fmtarg) \
		__attribute__((__format__ (__strftime__, fmtarg, 0)))
#else
#define __strfmonlike(fmtarg, firstvararg)
#define __strftimelike(fmtarg)
#endif

#ifndef UNIFDEF_DRIVERKIT
typedef	int		__darwin_nl_item;
#endif /* UNIFDEF_DRIVERKIT */
typedef	int		__darwin_wctrans_t;
#ifdef __LP64__
typedef	__uint32_t	__darwin_wctype_t;
#else /* !__LP64__ */
typedef	unsigned long	__darwin_wctype_t;
#endif /* __LP64__ */

#ifdef __WCHAR_MAX__
#define __DARWIN_WCHAR_MAX	__WCHAR_MAX__
#else /* ! __WCHAR_MAX__ */
#define __DARWIN_WCHAR_MAX	0x7fffffff
#endif /* __WCHAR_MAX__ */

#if __DARWIN_WCHAR_MAX > 0xffffU
#define __DARWIN_WCHAR_MIN	(-0x7fffffff - 1)
#else
#define __DARWIN_WCHAR_MIN	0
#endif
#define	__DARWIN_WEOF 	((__darwin_wint_t)-1)

#ifndef _FORTIFY_SOURCE
#  if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && ((__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0) < 1050)
#    define _FORTIFY_SOURCE 0
#  else
#    define _FORTIFY_SOURCE 2	/* on by default */
#  endif
#endif

#endif /* __TYPES_H_ */
