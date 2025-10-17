/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef _XLOCALE__CTYPE_H_
#define _XLOCALE__CTYPE_H_

#include <_ctype.h>
#include <_xlocale.h>

/*
 * Use inline functions if we are allowed to and the compiler supports them.
 */
#if !defined(_DONT_USE_CTYPE_INLINE_) && \
    (defined(_USE_CTYPE_INLINE_) || defined(__GNUC__) || defined(__cplusplus))

/* See comments in <machine/_type.h> about __darwin_ct_rune_t. */
__BEGIN_DECLS
unsigned long		___runetype_l(__darwin_ct_rune_t, locale_t);
__darwin_ct_rune_t	___tolower_l(__darwin_ct_rune_t, locale_t);
__darwin_ct_rune_t	___toupper_l(__darwin_ct_rune_t, locale_t);
__END_DECLS

//Begin-Libc
#ifdef __LIBC__
__DARWIN_CTYPE_inline int
__maskrune_l(__darwin_ct_rune_t _c, unsigned long _f, locale_t _l)
{
	/* _CurrentRuneLocale.__runetype is __uint32_t
	 * _f is unsigned long
	 * ___runetype_l(_c, _l) is unsigned long
	 * retval is int
	 */
	return (int)((_c < 0 || _c >= _CACHED_RUNES) ? (__uint32_t)___runetype_l(_c, _l) :
		__locale_ptr(_l)->__lc_ctype->_CurrentRuneLocale.__runetype[_c]) & (__uint32_t)_f;
}
#else /* !__LIBC__ */
//End-Libc
__BEGIN_DECLS
int             	__maskrune_l(__darwin_ct_rune_t, unsigned long, locale_t);
__END_DECLS
//Begin-Libc
#endif /* __LIBC__ */
//End-Libc

__DARWIN_CTYPE_inline int
__istype_l(__darwin_ct_rune_t _c, unsigned long _f, locale_t _l)
{
	return !!(isascii(_c) ? (_DefaultRuneLocale.__runetype[_c] & _f)
		: __maskrune_l(_c, _f, _l));
}

__DARWIN_CTYPE_inline __darwin_ct_rune_t
__toupper_l(__darwin_ct_rune_t _c, locale_t _l)
{
	return isascii(_c) ? _DefaultRuneLocale.__mapupper[_c]
		: ___toupper_l(_c, _l);
}

__DARWIN_CTYPE_inline __darwin_ct_rune_t
__tolower_l(__darwin_ct_rune_t _c, locale_t _l)
{
	return isascii(_c) ? _DefaultRuneLocale.__maplower[_c]
		: ___tolower_l(_c, _l);
}

__DARWIN_CTYPE_inline int
__wcwidth_l(__darwin_ct_rune_t _c, locale_t _l)
{
	unsigned int _x;

	if (_c == 0)
		return (0);
	_x = (unsigned int)__maskrune_l(_c, _CTYPE_SWM|_CTYPE_R, _l);
	if ((_x & _CTYPE_SWM) != 0)
		return ((_x & _CTYPE_SWM) >> _CTYPE_SWS);
	return ((_x & _CTYPE_R) != 0 ? 1 : -1);
}

#ifndef _EXTERNALIZE_CTYPE_INLINES_

__DARWIN_CTYPE_TOP_inline int
digittoint_l(int c, locale_t l)
{
	return (__maskrune_l(c, 0x0F, l));
}

__DARWIN_CTYPE_TOP_inline int
isalnum_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_A|_CTYPE_D, l));
}

__DARWIN_CTYPE_TOP_inline int
isalpha_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_A, l));
}

__DARWIN_CTYPE_TOP_inline int
isblank_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_B, l));
}

__DARWIN_CTYPE_TOP_inline int
iscntrl_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_C, l));
}

__DARWIN_CTYPE_TOP_inline int
isdigit_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_D, l));
}

__DARWIN_CTYPE_TOP_inline int
isgraph_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_G, l));
}

__DARWIN_CTYPE_TOP_inline int
ishexnumber_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_X, l));
}

__DARWIN_CTYPE_TOP_inline int
isideogram_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_I, l));
}

__DARWIN_CTYPE_TOP_inline int
islower_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_L, l));
}

__DARWIN_CTYPE_TOP_inline int
isnumber_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_D, l));
}

__DARWIN_CTYPE_TOP_inline int
isphonogram_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_Q, l));
}

__DARWIN_CTYPE_TOP_inline int
isprint_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_R, l));
}

__DARWIN_CTYPE_TOP_inline int
ispunct_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_P, l));
}

__DARWIN_CTYPE_TOP_inline int
isrune_l(int c, locale_t l)
{
	return (__istype_l(c, 0xFFFFFFF0L, l));
}

__DARWIN_CTYPE_TOP_inline int
isspace_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_S, l));
}

__DARWIN_CTYPE_TOP_inline int
isspecial_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_T, l));
}

__DARWIN_CTYPE_TOP_inline int
isupper_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_U, l));
}

__DARWIN_CTYPE_TOP_inline int
isxdigit_l(int c, locale_t l)
{
	return (__istype_l(c, _CTYPE_X, l));
}

__DARWIN_CTYPE_TOP_inline int
tolower_l(int c, locale_t l)
{
        return (__tolower_l(c, l));
}

__DARWIN_CTYPE_TOP_inline int
toupper_l(int c, locale_t l)
{
        return (__toupper_l(c, l));
}
#endif /* _EXTERNALIZE_CTYPE_INLINES_ */

#else /* not using inlines */

__BEGIN_DECLS
int     digittoint_l(int, locale_t);
int     isalnum_l(int, locale_t);
int     isalpha_l(int, locale_t);
int     isblank_l(int, locale_t);
int     iscntrl_l(int, locale_t);
int     isdigit_l(int, locale_t);
int     isgraph_l(int, locale_t);
int     ishexnumber_l(int, locale_t);
int     isideogram_l(int, locale_t);
int     islower_l(int, locale_t);
int     isnumber_l(int, locale_t);
int     isphonogram_l(int, locale_t);
int     isprint_l(int, locale_t);
int     ispunct_l(int, locale_t);
int     isrune_l(int, locale_t);
int     isspace_l(int, locale_t);
int     isspecial_l(int, locale_t);
int     isupper_l(int, locale_t);
int     isxdigit_l(int, locale_t);
int     tolower_l(int, locale_t);
int     toupper_l(int, locale_t);
__END_DECLS
#endif /* using inlines */

#endif /* _XLOCALE__CTYPE_H_ */
