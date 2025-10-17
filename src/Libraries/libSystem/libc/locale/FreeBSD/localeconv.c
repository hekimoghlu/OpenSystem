/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)localeconv.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/localeconv.c,v 1.14 2007/12/12 07:43:23 phantom Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <locale.h>

#include "lmonetary.h"
#include "lnumeric.h"

#ifdef __APPLE_PR3417676_HACK__
/*------------------------------------------------------------------------
 * PR-3417676: We need to provide a way to force "C" locale style number
 * formatting independent of the locale setting.  We provide private
 * routines to get and set a flag that tells localeconv() to either return
 * a "C" struct lconv, or the one dependent on the actual locale.
 *------------------------------------------------------------------------*/
static char empty[] = "";
static char numempty[] = { CHAR_MAX, '\0' };

/*
 * Default (C) locale conversion.
 */
static struct lconv _C_lconv = {
	".",			/* decimal_point */
	empty,			/* thousands_sep */
	numempty,		/* grouping */
	empty,			/* int_curr_symbol */
	empty,			/* currency_symbol */
	empty,			/* mon_decimal_point */
	empty,			/* mon_thousands_sep */
	numempty,		/* mon_grouping */
	empty,			/* positive_sign */
	empty,			/* negative_sign */
	CHAR_MAX,		/* int_frac_digits */
	CHAR_MAX,		/* frac_digits */
	CHAR_MAX,		/* p_cs_precedes */
	CHAR_MAX,		/* p_sep_by_space */
	CHAR_MAX,		/* n_cs_precedes */
	CHAR_MAX,		/* n_sep_by_space */
	CHAR_MAX,		/* p_sign_posn */
	CHAR_MAX,		/* n_sign_posn */
	CHAR_MAX,		/* int_p_cs_precedes */
	CHAR_MAX,		/* int_n_cs_precedes */
	CHAR_MAX,		/* int_p_sep_by_space */
	CHAR_MAX,		/* int_n_sep_by_space */
	CHAR_MAX,		/* int_p_sign_posn */
	CHAR_MAX,		/* int_n_sign_posn */
};
static int _onlyClocaleconv = 0;

int
__getonlyClocaleconv(void)
{
    return _onlyClocaleconv;
}

int
__setonlyClocaleconv(int val)
{
    int prev = _onlyClocaleconv;

    _onlyClocaleconv = val;
    return prev;
}
#endif /* __APPLE_PR3417676_HACK__ */

/* 
 * The localeconv() function constructs a struct lconv from the current
 * monetary and numeric locales.
 *
 * Because localeconv() may be called many times (especially by library
 * routines like printf() & strtod()), the approprate members of the 
 * lconv structure are computed only when the monetary or numeric 
 * locale has been changed.
 */

/*
 * Return the current locale conversion.
 */
struct lconv *
localeconv_l(locale_t loc)
{
    NORMALIZE_LOCALE(loc);

    if (loc->__mlocale_changed) {
      XL_LOCK(loc);
      if (loc->__mlocale_changed) {
	/* LC_MONETARY part */
        struct lc_monetary_T * mptr; 
	struct lconv *lc = &loc->__lc_localeconv;

#define M_ASSIGN_STR(NAME) (lc->NAME = (char*)mptr->NAME)
#define M_ASSIGN_CHAR(NAME) (lc->NAME = mptr->NAME[0])

	mptr = __get_current_monetary_locale(loc);
	M_ASSIGN_STR(int_curr_symbol);
	M_ASSIGN_STR(currency_symbol);
	M_ASSIGN_STR(mon_decimal_point);
	M_ASSIGN_STR(mon_thousands_sep);
	M_ASSIGN_STR(mon_grouping);
	M_ASSIGN_STR(positive_sign);
	M_ASSIGN_STR(negative_sign);
	M_ASSIGN_CHAR(int_frac_digits);
	M_ASSIGN_CHAR(frac_digits);
	M_ASSIGN_CHAR(p_cs_precedes);
	M_ASSIGN_CHAR(p_sep_by_space);
	M_ASSIGN_CHAR(n_cs_precedes);
	M_ASSIGN_CHAR(n_sep_by_space);
	M_ASSIGN_CHAR(p_sign_posn);
	M_ASSIGN_CHAR(n_sign_posn);
	M_ASSIGN_CHAR(int_p_cs_precedes);
	M_ASSIGN_CHAR(int_n_cs_precedes);
	M_ASSIGN_CHAR(int_p_sep_by_space);
	M_ASSIGN_CHAR(int_n_sep_by_space);
	M_ASSIGN_CHAR(int_p_sign_posn);
	M_ASSIGN_CHAR(int_n_sign_posn);
	loc->__mlocale_changed = 0;
      }
      XL_UNLOCK(loc);
    }

    if (loc->__nlocale_changed) {
      XL_LOCK(loc);
      if (loc->__nlocale_changed) {
	/* LC_NUMERIC part */
        struct lc_numeric_T * nptr; 
	struct lconv *lc = &loc->__lc_localeconv;

#define N_ASSIGN_STR(NAME) (lc->NAME = (char*)nptr->NAME)

	nptr = __get_current_numeric_locale(loc);
	N_ASSIGN_STR(decimal_point);
	N_ASSIGN_STR(thousands_sep);
	N_ASSIGN_STR(grouping);
	loc->__nlocale_changed = 0;
      }
      XL_UNLOCK(loc);
    }

    return &loc->__lc_localeconv;
}

/*
 * Return the current locale conversion.
 */
struct lconv *
localeconv()
{
#ifdef __APPLE_PR3417676_HACK__
    /*--------------------------------------------------------------------
     * If _onlyClocaleconv is non-zero, just return __lconv, which is a "C"
     * struct lconv *.  Otherwise, do the normal thing.
     *--------------------------------------------------------------------*/
    if (_onlyClocaleconv)
	return &_C_lconv;
#endif /* __APPLE_PR3417676_HACK__ */
    return localeconv_l(__current_locale());
}
