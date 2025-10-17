/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include <config.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include <ctype.h>
#include "misc-proto.h"

RCSID("$Id$");


#define	LOWER(x) (isupper(x) ? tolower(x) : (x))
/*
 * The prefix function returns 0 if *s1 is not a prefix
 * of *s2.  If *s1 exactly matches *s2, the negative of
 * the length is returned.  If *s1 is a prefix of *s2,
 * the length of *s1 is returned.
 */

int
isprefix(char *s1, char *s2)
{
    char *os1;
    char c1, c2;

    if (*s1 == '\0')
	return(-1);
    os1 = s1;
    c1 = *s1;
    c2 = *s2;
    while (tolower((unsigned char)c1) == tolower((unsigned char)c2)) {
	if (c1 == '\0')
	    break;
	c1 = *++s1;
	c2 = *++s2;
    }
    return(*s1 ? 0 : (*s2 ? (s1 - os1) : (os1 - s1)));
}

static char *ambiguous;		/* special return value for command routines */

char **
genget(char *name, char **table, int stlen)
     /* name to match */
     /* name entry in table */

{
    char **c, **found;
    int n;

    if (name == 0)
	return 0;

    found = 0;
    for (c = table; *c != 0; c = (char **)((char *)c + stlen)) {
	if ((n = isprefix(name, *c)) == 0)
	    continue;
	if (n < 0)		/* exact match */
	    return(c);
	if (found)
	    return(&ambiguous);
	found = c;
    }
    return(found);
}

/*
 * Function call version of Ambiguous()
 */
int
Ambiguous(void *s)
{
    return((char **)s == &ambiguous);
}
