/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
/*
 * LC_MONETARY database generation routines for localedef.
 */
#include <sys/cdefs.h>

#ifdef __APPLE__
#include <sys/param.h>

#include <stddef.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include "localedef.h"
#include "parser.h"
#include "lmonetary.h"

static struct lc_monetary_T mon;

#ifdef __APPLE__
#define MON_FIELD(name)	offsetof(struct lc_monetary_T, name)

/*
 * We only need to provide default values for numeric fields, which should print
 * -1 instead of the empty string.
 */
static struct monetary_field_init {
	size_t		 offset;
	const char	*value;
} monetary_init[] = {
	{ MON_FIELD(int_frac_digits), "-1" },
	{ MON_FIELD(frac_digits), "-1" },
	{ MON_FIELD(p_cs_precedes), "-1" },
	{ MON_FIELD(p_sep_by_space), "-1" },
	{ MON_FIELD(n_cs_precedes), "-1" },
	{ MON_FIELD(n_sep_by_space), "-1" },
	{ MON_FIELD(p_sign_posn), "-1" },
	{ MON_FIELD(n_sign_posn), "-1" },
	{ MON_FIELD(int_p_cs_precedes), "-1" },
	{ MON_FIELD(int_p_sep_by_space), "-1" },
	{ MON_FIELD(int_n_cs_precedes), "-1" },
	{ MON_FIELD(int_n_sep_by_space), "-1" },
	{ MON_FIELD(int_p_sign_posn), "-1" },
	{ MON_FIELD(int_n_sign_posn), "-1" },
};
#endif	/* __APPLE__ */

void
init_monetary(void)
{
	(void) memset(&mon, 0, sizeof (mon));
}

void
add_monetary_str(wchar_t *wcs)
{
	char *str;

	if ((str = to_mb_string(wcs)) == NULL) {
		INTERR;
		return;
	}
	free(wcs);
	switch (last_kw) {
	case T_INT_CURR_SYMBOL:
		mon.int_curr_symbol = str;
		break;
	case T_CURRENCY_SYMBOL:
		mon.currency_symbol = str;
		break;
	case T_MON_DECIMAL_POINT:
		mon.mon_decimal_point = str;
		break;
	case T_MON_THOUSANDS_SEP:
		mon.mon_thousands_sep = str;
		break;
	case T_POSITIVE_SIGN:
		mon.positive_sign = str;
		break;
	case T_NEGATIVE_SIGN:
		mon.negative_sign = str;
		break;
	default:
		free(str);
		INTERR;
		break;
	}
}

void
add_monetary_num(int n)
{
	char *str = NULL;

	(void) asprintf(&str, "%d", n);
	if (str == NULL) {
		fprintf(stderr, "out of memory\n");
		return;
	}

	switch (last_kw) {
	case T_INT_FRAC_DIGITS:
		mon.int_frac_digits = str;
		break;
	case T_FRAC_DIGITS:
		mon.frac_digits = str;
		break;
	case T_P_CS_PRECEDES:
		mon.p_cs_precedes = str;
		break;
	case T_P_SEP_BY_SPACE:
		mon.p_sep_by_space = str;
		break;
	case T_N_CS_PRECEDES:
		mon.n_cs_precedes = str;
		break;
	case T_N_SEP_BY_SPACE:
		mon.n_sep_by_space = str;
		break;
	case T_P_SIGN_POSN:
		mon.p_sign_posn = str;
		break;
	case T_N_SIGN_POSN:
		mon.n_sign_posn = str;
		break;
	case T_INT_P_CS_PRECEDES:
		mon.int_p_cs_precedes = str;
		break;
	case T_INT_N_CS_PRECEDES:
		mon.int_n_cs_precedes = str;
		break;
	case T_INT_P_SEP_BY_SPACE:
		mon.int_p_sep_by_space = str;
		break;
	case T_INT_N_SEP_BY_SPACE:
		mon.int_n_sep_by_space = str;
		break;
	case T_INT_P_SIGN_POSN:
		mon.int_p_sign_posn = str;
		break;
	case T_INT_N_SIGN_POSN:
		mon.int_n_sign_posn = str;
		break;
	case T_MON_GROUPING:
		mon.mon_grouping = str;
		break;
	default:
		INTERR;
		break;
	}
}

void
reset_monetary_group(void)
{
	free((char *)mon.mon_grouping);
	mon.mon_grouping = NULL;
}

void
add_monetary_group(int n)
{
	char *s = NULL;

	if (mon.mon_grouping == NULL) {
		(void) asprintf(&s, "%d", n);
	} else {
		(void) asprintf(&s, "%s;%d", mon.mon_grouping, n);
	}
	if (s == NULL)
		fprintf(stderr, "out of memory\n");

	free((char *)mon.mon_grouping);
	mon.mon_grouping = s;
}

void
dump_monetary(void)
{
	FILE *f;

#ifdef __APPLE__
	for (size_t i = 0; i < nitems(monetary_init); i++) {
		struct monetary_field_init *initf = &monetary_init[i];
		const char **field;

		field = (const char **)(((unsigned char *)&mon) + initf->offset);
		if (*field == NULL)
			*field = initf->value;
	}
#endif

	if ((f = open_category()) == NULL) {
		return;
	}

	if ((putl_category(mon.int_curr_symbol, f) == EOF) ||
	    (putl_category(mon.currency_symbol, f) == EOF) ||
	    (putl_category(mon.mon_decimal_point, f) == EOF) ||
	    (putl_category(mon.mon_thousands_sep, f) == EOF) ||
	    (putl_category(mon.mon_grouping, f) == EOF) ||
	    (putl_category(mon.positive_sign, f) == EOF) ||
	    (putl_category(mon.negative_sign, f) == EOF) ||
	    (putl_category(mon.int_frac_digits, f) == EOF) ||
	    (putl_category(mon.frac_digits, f) == EOF) ||
	    (putl_category(mon.p_cs_precedes, f) == EOF) ||
	    (putl_category(mon.p_sep_by_space, f) == EOF) ||
	    (putl_category(mon.n_cs_precedes, f) == EOF) ||
	    (putl_category(mon.n_sep_by_space, f) == EOF) ||
	    (putl_category(mon.p_sign_posn, f) == EOF) ||
	    (putl_category(mon.n_sign_posn, f) == EOF) ||
	    (putl_category(mon.int_p_cs_precedes, f) == EOF) ||
	    (putl_category(mon.int_n_cs_precedes, f) == EOF) ||
	    (putl_category(mon.int_p_sep_by_space, f) == EOF) ||
	    (putl_category(mon.int_n_sep_by_space, f) == EOF) ||
	    (putl_category(mon.int_p_sign_posn, f) == EOF) ||
	    (putl_category(mon.int_n_sign_posn, f) == EOF)) {
		return;
	}
	close_category(f);
}
