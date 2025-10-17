/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
 * LC_NUMERIC database generation routines for localedef.
 */
#include <sys/cdefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include "localedef.h"
#include "parser.h"
#include "lnumeric.h"

static struct lc_numeric_T numeric;

void
init_numeric(void)
{
	(void) memset(&numeric, 0, sizeof (numeric));
}

void
add_numeric_str(wchar_t *wcs)
{
	char *str;

	if ((str = to_mb_string(wcs)) == NULL) {
		INTERR;
		return;
	}
	free(wcs);

	switch (last_kw) {
	case T_DECIMAL_POINT:
		numeric.decimal_point = str;
		break;
	case T_THOUSANDS_SEP:
		numeric.thousands_sep = str;
		break;
	default:
		free(str);
		INTERR;
		break;
	}
}

void
reset_numeric_group(void)
{
	free((char *)numeric.grouping);
	numeric.grouping = NULL;
}

void
add_numeric_group(int n)
{
	char *s;

	if (numeric.grouping == NULL) {
		(void) asprintf(&s, "%d", n);
	} else {
		(void) asprintf(&s, "%s;%d", numeric.grouping, n);
	}
	if (s == NULL)
		fprintf(stderr, "out of memory\n");

	free((char *)numeric.grouping);
	numeric.grouping = s;
}

void
dump_numeric(void)
{
	FILE *f;

	if ((f = open_category()) == NULL) {
		return;
	}

	if ((putl_category(numeric.decimal_point, f) == EOF) ||
	    (putl_category(numeric.thousands_sep, f) == EOF) ||
	    (putl_category(numeric.grouping, f) == EOF)) {
		return;
	}
	close_category(f);
}
