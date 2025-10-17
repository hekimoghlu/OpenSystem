/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#ifndef __LOCALE_H_
#define __LOCALE_H_

#include <sys/cdefs.h>
#include <_bounds.h>
#include <_types.h>

_LIBC_SINGLE_BY_DEFAULT()

struct lconv {
	char	*_LIBC_CSTR decimal_point;
	char	*_LIBC_CSTR thousands_sep;
	char	*_LIBC_CSTR grouping;
	char	*_LIBC_CSTR int_curr_symbol;
	char	*_LIBC_CSTR currency_symbol;
	char	*_LIBC_CSTR mon_decimal_point;
	char	*_LIBC_CSTR mon_thousands_sep;
	char	*_LIBC_CSTR mon_grouping;
	char	*_LIBC_CSTR positive_sign;
	char	*_LIBC_CSTR negative_sign;
	char	int_frac_digits;
	char	frac_digits;
	char	p_cs_precedes;
	char	p_sep_by_space;
	char	n_cs_precedes;
	char	n_sep_by_space;
	char	p_sign_posn;
	char	n_sign_posn;
	char	int_p_cs_precedes;
	char	int_n_cs_precedes;
	char	int_p_sep_by_space;
	char	int_n_sep_by_space;
	char	int_p_sign_posn;
	char	int_n_sign_posn;
};

#include <sys/_types/_null.h>

#define LC_ALL_MASK			(  LC_COLLATE_MASK \
					 | LC_CTYPE_MASK \
					 | LC_MESSAGES_MASK \
					 | LC_MONETARY_MASK \
					 | LC_NUMERIC_MASK \
					 | LC_TIME_MASK )
#define LC_COLLATE_MASK			(1 << 0)
#define LC_CTYPE_MASK			(1 << 1)
#define LC_MESSAGES_MASK		(1 << 2)
#define LC_MONETARY_MASK		(1 << 3)
#define LC_NUMERIC_MASK			(1 << 4)
#define LC_TIME_MASK			(1 << 5)

#define _LC_NUM_MASK			6
#define _LC_LAST_MASK			(1 << (_LC_NUM_MASK - 1))

#define LC_GLOBAL_LOCALE		((locale_t)-1)
#define LC_C_LOCALE				((locale_t)NULL)

#include <_types/_locale_t.h>

__BEGIN_DECLS
locale_t	duplocale(locale_t);
int		freelocale(locale_t);
struct lconv	*localeconv(void);
locale_t	newlocale(int, __const char *, locale_t);
locale_t	uselocale(locale_t);
__END_DECLS

#endif /* __LOCALE_H_ */
