/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
#ifndef _BITS_WCTYPE_H_
#define _BITS_WCTYPE_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

typedef __WINT_TYPE__ wint_t;

#define WEOF __BIONIC_CAST(static_cast, wint_t, -1)

int iswalnum(wint_t __wc);
int iswalpha(wint_t __wc);
int iswblank(wint_t __wc);
int iswcntrl(wint_t __wc);
int iswdigit(wint_t __wc);
int iswgraph(wint_t __wc);
int iswlower(wint_t __wc);
int iswprint(wint_t __wc);
int iswpunct(wint_t __wc);
int iswspace(wint_t __wc);
int iswupper(wint_t __wc);
int iswxdigit(wint_t __wc);

wint_t towlower(wint_t __wc);
wint_t towupper(wint_t __wc);

typedef long wctype_t;
wctype_t wctype(const char* _Nonnull __name);
int iswctype(wint_t __wc, wctype_t __type);

typedef const void* wctrans_t;

#if __BIONIC_AVAILABILITY_GUARD(26)
wint_t towctrans(wint_t __wc, wctrans_t _Nonnull __transform) __INTRODUCED_IN(26);
wctrans_t _Nullable wctrans(const char* _Nonnull __name) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS

#endif
