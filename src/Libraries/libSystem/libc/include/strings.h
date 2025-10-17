/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
/*-
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)strings.h	8.1 (Berkeley) 6/2/93
 */

#ifndef _STRINGS_H_
#define _STRINGS_H_

#include <_types.h>

#include <sys/cdefs.h>
#include <Availability.h>
#include <sys/_types/_size_t.h>

__BEGIN_DECLS
/* Removed in Issue 7 */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
int	 bcmp(const void *, const void *, size_t) __POSIX_C_DEPRECATED(200112L);
void	 bcopy(const void *, void *, size_t) __POSIX_C_DEPRECATED(200112L);
void	 bzero(void *, size_t) __POSIX_C_DEPRECATED(200112L);
char	*index(const char *, int) __POSIX_C_DEPRECATED(200112L);
char	*rindex(const char *, int) __POSIX_C_DEPRECATED(200112L);
#endif

int	 ffs(int);
int	 strcasecmp(const char *, const char *);
int	 strncasecmp(const char *, const char *, size_t);
__END_DECLS

/* Darwin extensions */
#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL
__BEGIN_DECLS
int	 ffsl(long) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
int	 ffsll(long long) __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
int	 fls(int) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
int	 flsl(long) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
int	 flsll(long long) __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
__END_DECLS

#include <string.h>
#endif

#if defined (__GNUC__) && _FORTIFY_SOURCE > 0 && !defined (__cplusplus)
/* Security checking functions.  */
#include <secure/_strings.h>
#endif

#endif  /* _STRINGS_H_ */

