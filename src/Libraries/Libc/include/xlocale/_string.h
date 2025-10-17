/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#ifndef _XLOCALE__STRING_H_
#define _XLOCALE__STRING_H_

#include <sys/cdefs.h>
#include <_bounds.h>
#include <sys/_types/_size_t.h>
#include <__xlocale.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS
int	 strcoll_l(const char *, const char *, locale_t);
size_t	 strxfrm_l(char *_LIBC_COUNT(__n), const char *, size_t __n, locale_t);
int	 strcasecmp_l(const char *, const char *, locale_t);
char    *strcasestr_l(const char *, const char *, locale_t);
int	 strncasecmp_l(const char *_LIBC_UNSAFE_INDEXABLE,
        const char *_LIBC_UNSAFE_INDEXABLE, size_t, locale_t);
__END_DECLS

#endif /* _XLOCALE__STRING_H_ */
