/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
/****************************************************************************
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/
#include <progs.priv.h>
#include <string.h>

#include <transform.h>

MODULE_ID("$Id: transform.c,v 1.3 2011/05/14 22:41:17 tom Exp $")

#ifdef SUFFIX_IGNORED
static void
trim_suffix(const char *a, size_t *len)
{
    const char ignore[] = SUFFIX_IGNORED;

    if (sizeof(ignore) != 0) {
	bool trim = FALSE;
	size_t need = (sizeof(ignore) - 1);

	if (*len > need) {
	    size_t first = *len - need;
	    size_t n;
	    trim = TRUE;
	    for (n = first; n < *len; ++n) {
		if (tolower(UChar(a[n])) != tolower(UChar(ignore[n - first]))) {
		    trim = FALSE;
		    break;
		}
	    }
	    if (trim) {
		*len -= need;
	    }
	}
    }
}
#else
#define trim_suffix(a, len)	/* nothing */
#endif

bool
same_program(const char *a, const char *b)
{
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);

    trim_suffix(a, &len_a);
    trim_suffix(b, &len_b);

    return (len_a == len_b) && (strncmp(a, b, len_a) == 0);
}
