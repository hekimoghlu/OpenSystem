/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#ifndef _XLOCALE__WCTYPE_H_
#define _XLOCALE__WCTYPE_H_

#include <_bounds.h>
#include <___wctype.h>
#include <_types/_wctrans_t.h>
#include <xlocale/_ctype.h>

_LIBC_SINGLE_BY_DEFAULT()

#if !defined(_DONT_USE_CTYPE_INLINE_) && \
    (defined(_USE_CTYPE_INLINE_) || defined(__GNUC__) || defined(__cplusplus))

__DARWIN_WCTYPE_TOP_inline int
iswblank_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_B, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswhexnumber_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_X, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswideogram_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_I, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswnumber_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_D, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswphonogram_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_Q, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswrune_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, 0xFFFFFFF0L, _l));
}

__DARWIN_WCTYPE_TOP_inline int
iswspecial_l(wint_t _wc, locale_t _l)
{
	return (__istype_l(_wc, _CTYPE_T, _l));
}

#else /* not using inlines */

__BEGIN_DECLS
int	iswblank_l(wint_t, locale_t);
wint_t	iswhexnumber_l(wint_t, locale_t);
wint_t	iswideogram_l(wint_t, locale_t);
wint_t	iswnumber_l(wint_t, locale_t);
wint_t	iswphonogram_l(wint_t, locale_t);
wint_t	iswrune_l(wint_t, locale_t);
wint_t	iswspecial_l(wint_t, locale_t);
__END_DECLS

#endif /* using inlines */

__BEGIN_DECLS
wint_t	nextwctype_l(wint_t, wctype_t, locale_t);
wint_t	towctrans_l(wint_t, wctrans_t, locale_t);
wctrans_t
	wctrans_l(const char *, locale_t);
__END_DECLS

#endif /* _XLOCALE__WCTYPE_H_ */
