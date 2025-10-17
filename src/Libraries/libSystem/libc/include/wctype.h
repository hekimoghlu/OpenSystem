/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#define	_WCTYPE_H_

#include <sys/cdefs.h>
#include <_types.h>
#include <_types/_wctrans_t.h>

//Begin-Libc
/*
 * _EXTERNALIZE_WCTYPE_INLINES_TOP_ is defined in locale/iswctype.c to tell us
 * to generate code for extern versions of all top-level inline functions.
 */
#ifdef _EXTERNALIZE_WCTYPE_INLINES_TOP_
#define _USE_CTYPE_INLINE_
#define __DARWIN_WCTYPE_TOP_inline
#else /* !_EXTERNALIZE_WCTYPE_INLINES_TOP_ */
//End-Libc
#define __DARWIN_WCTYPE_TOP_inline	__header_inline
//Begin-Libc
#endif /* _EXTERNALIZE_WCTYPE_INLINES_TOP_ */
//End-Libc

#include <_wctype.h>
#include <ctype.h>

/*
 * Use inline functions if we are allowed to and the compiler supports them.
 */
#if !defined(_DONT_USE_CTYPE_INLINE_) && \
    (defined(_USE_CTYPE_INLINE_) || defined(__GNUC__) || defined(__cplusplus))

__DARWIN_WCTYPE_TOP_inline int
iswblank(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_B));
}

#if !defined(_ANSI_SOURCE)
__DARWIN_WCTYPE_TOP_inline int
iswascii(wint_t _wc)
{
	return ((_wc & ~0x7F) == 0);
}

__DARWIN_WCTYPE_TOP_inline int
iswhexnumber(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_X));
}

__DARWIN_WCTYPE_TOP_inline int
iswideogram(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_I));
}

__DARWIN_WCTYPE_TOP_inline int
iswnumber(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_D));
}

__DARWIN_WCTYPE_TOP_inline int
iswphonogram(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_Q));
}

__DARWIN_WCTYPE_TOP_inline int
iswrune(wint_t _wc)
{
	return (__istype(_wc, 0xFFFFFFF0L));
}

__DARWIN_WCTYPE_TOP_inline int
iswspecial(wint_t _wc)
{
	return (__istype(_wc, _CTYPE_T));
}
#endif /* !_ANSI_SOURCE */

#else /* not using inlines */

__BEGIN_DECLS
int	iswblank(wint_t);

#if !defined(_ANSI_SOURCE)
wint_t	iswascii(wint_t);
wint_t	iswhexnumber(wint_t);
wint_t	iswideogram(wint_t);
wint_t	iswnumber(wint_t);
wint_t	iswphonogram(wint_t);
wint_t	iswrune(wint_t);
wint_t	iswspecial(wint_t);
#endif
__END_DECLS

#endif /* using inlines */

__BEGIN_DECLS
#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
wint_t	nextwctype(wint_t, wctype_t);
#endif
wint_t	towctrans(wint_t, wctrans_t);
wctrans_t
	wctrans(const char *);
__END_DECLS

#ifdef _USE_EXTENDED_LOCALES_
#include <xlocale/_wctype.h>
#endif /* _USE_EXTENDED_LOCALES_ */

#endif		/* _WCTYPE_H_ */
