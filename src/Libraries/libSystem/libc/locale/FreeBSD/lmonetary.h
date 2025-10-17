/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#ifndef _LMONETARY_H_
#define	_LMONETARY_H_

#include <xlocale.h>

struct lc_monetary_T {
	const char	*int_curr_symbol;
	const char	*currency_symbol;
	const char	*mon_decimal_point;
	const char	*mon_thousands_sep;
	const char	*mon_grouping;
	const char	*positive_sign;
	const char	*negative_sign;
	const char	*int_frac_digits;
	const char	*frac_digits;
	const char	*p_cs_precedes;
	const char	*p_sep_by_space;
	const char	*n_cs_precedes;
	const char	*n_sep_by_space;
	const char	*p_sign_posn;
	const char	*n_sign_posn;
	const char	*int_p_cs_precedes;
	const char	*int_n_cs_precedes;
	const char	*int_p_sep_by_space;
	const char	*int_n_sep_by_space;
	const char	*int_p_sign_posn;
	const char	*int_n_sign_posn;
};

struct lc_monetary_T *__get_current_monetary_locale(locale_t);
int	__monetary_load_locale(const char *, locale_t);

#endif /* !_LMONETARY_H_ */
