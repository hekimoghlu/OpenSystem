/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#ifndef _XLOCALE__REGEX_H_
#define _XLOCALE__REGEX_H_

#ifndef _REGEX_H_
#include <_regex.h>
#include <__xlocale.h>
#endif

#include <_bounds.h>
_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS

int	regcomp_l(regex_t * __restrict, const char * __restrict, int,
	    locale_t __restrict)
	    __OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_NA);

#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL

int	regncomp_l(regex_t * __restrict, const char * __restrict _LIBC_COUNT(__len),
	    size_t __len, int, locale_t __restrict)
	    __OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_NA);
int	regwcomp_l(regex_t * __restrict, const wchar_t * __restrict,
	    int, locale_t __restrict)
	    __OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_NA);
int	regwnexec_l(const regex_t * __restrict, const wchar_t * __restrict _LIBC_COUNT(__len),
	    size_t __len, size_t __nmatch, regmatch_t __pmatch[ __restrict _LIBC_COUNT(__nmatch)], int,
	    locale_t __restrict)
	    __OSX_AVAILABLE_STARTING(__MAC_10_8, __IPHONE_NA);

#endif /* __DARWIN_C_LEVEL >= __DARWIN_C_FULL */

__END_DECLS

#endif /* _XLOCALE__REGEX_H_ */
