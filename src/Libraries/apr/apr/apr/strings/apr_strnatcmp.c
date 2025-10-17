/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include <ctype.h>
#include <string.h>
#include "apr_strings.h"
#include "apr_lib.h"            /* for apr_is*() */

#if defined(__GNUC__)
#  define UNUSED __attribute__((__unused__))
#else
#  define UNUSED
#endif

/* based on "strnatcmp.c,v 1.6 2000/04/20 07:30:11 mbp Exp $" */

static int
compare_right(char const *a, char const *b)
{
     int bias = 0;
     
     /* The longest run of digits wins.  That aside, the greatest
	value wins, but we can't know that it will until we've scanned
	both numbers to know that they have the same magnitude, so we
	remember it in BIAS. */
     for (;; a++, b++) {
	  if (!apr_isdigit(*a)  &&  !apr_isdigit(*b))
	       break;
	  else if (!apr_isdigit(*a))
	       return -1;
	  else if (!apr_isdigit(*b))
	       return +1;
	  else if (*a < *b) {
	       if (!bias)
		    bias = -1;
	  } else if (*a > *b) {
	       if (!bias)
		    bias = +1;
	  } else if (!*a  &&  !*b)
	       break;
     }

     return bias;
}


static int
compare_left(char const *a, char const *b)
{
     /* Compare two left-aligned numbers: the first to have a
        different value wins. */
     for (;; a++, b++) {
	  if (!apr_isdigit(*a)  &&  !apr_isdigit(*b))
	       break;
	  else if (!apr_isdigit(*a))
	       return -1;
	  else if (!apr_isdigit(*b))
	       return +1;
	  else if (*a < *b)
	       return -1;
	  else if (*a > *b)
	       return +1;
     }
	  
     return 0;
}


static int strnatcmp0(char const *a, char const *b, int fold_case)
{
     int ai, bi;
     char ca, cb;
     int fractional, result;
     ai = bi = 0;
     while (1) {
	  ca = a[ai]; cb = b[bi];

	  /* skip over leading spaces or zeros */
	  while (apr_isspace(ca))
	       ca = a[++ai];

	  while (apr_isspace(cb))
	       cb = b[++bi];

	  /* process run of digits */
	  if (apr_isdigit(ca)  &&  apr_isdigit(cb)) {
	       fractional = (ca == '0' || cb == '0');

	       if (fractional) {
		    if ((result = compare_left(a+ai, b+bi)) != 0)
			 return result;
	       } else {
		    if ((result = compare_right(a+ai, b+bi)) != 0)
			 return result;
	       }
	  }

	  if (!ca && !cb) {
	       /* The strings compare the same.  Perhaps the caller
                  will want to call strcmp to break the tie. */
	       return 0;
	  }

	  if (fold_case) {
	       ca = apr_toupper(ca);
	       cb = apr_toupper(cb);
	  }
	  
	  if (ca < cb)
	       return -1;
	  else if (ca > cb)
	       return +1;

	  ++ai; ++bi;
     }
}



APR_DECLARE(int) apr_strnatcmp(char const *a, char const *b)
{
     return strnatcmp0(a, b, 0);
}


/* Compare, recognizing numeric string and ignoring case. */
APR_DECLARE(int) apr_strnatcasecmp(char const *a, char const *b)
{
     return strnatcmp0(a, b, 1);
}
