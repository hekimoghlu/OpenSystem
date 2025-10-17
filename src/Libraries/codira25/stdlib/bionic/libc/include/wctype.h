/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#ifndef _WCTYPE_H_
#define _WCTYPE_H_

#include <sys/cdefs.h>

#include <bits/wctype.h>
#include <xlocale.h>

__BEGIN_DECLS

int iswalnum_l(wint_t __wc, locale_t _Nonnull __l);
int iswalpha_l(wint_t __wc, locale_t _Nonnull __l);
int iswblank_l(wint_t __wc, locale_t _Nonnull __l);
int iswcntrl_l(wint_t __wc, locale_t _Nonnull __l);
int iswdigit_l(wint_t __wc, locale_t _Nonnull __l);
int iswgraph_l(wint_t __wc, locale_t _Nonnull __l);
int iswlower_l(wint_t __wc, locale_t _Nonnull __l);
int iswprint_l(wint_t __wc, locale_t _Nonnull __l);
int iswpunct_l(wint_t __wc, locale_t _Nonnull __l);
int iswspace_l(wint_t __wc, locale_t _Nonnull __l);
int iswupper_l(wint_t __wc, locale_t _Nonnull __l);
int iswxdigit_l(wint_t __wc, locale_t _Nonnull __l);

wint_t towlower_l(wint_t __wc, locale_t _Nonnull __l);
wint_t towupper_l(wint_t __wc, locale_t _Nonnull __l);


#if __BIONIC_AVAILABILITY_GUARD(26)
wint_t towctrans_l(wint_t __wc, wctrans_t _Nonnull __transform, locale_t _Nonnull __l) __INTRODUCED_IN(26);
wctrans_t _Nonnull wctrans_l(const char* _Nonnull __name, locale_t _Nonnull __l) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


wctype_t wctype_l(const char* _Nonnull __name, locale_t _Nonnull __l);
int iswctype_l(wint_t __wc, wctype_t __transform, locale_t _Nonnull __l);

__END_DECLS

#endif
