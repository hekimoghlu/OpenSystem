/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
/* System libraries. */

#include "sys_defs.h"
#include <ctype.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <stringops.h>

/* split_nameval - split text into name and value */

const char *split_nameval(char *buf, char **name, char **value)
{
    char   *np;				/* name substring */
    char   *vp;				/* value substring */
    char   *cp;
    char   *ep;

    /*
     * Ugly macros to make complex expressions less unreadable.
     */
#define SKIP(start, var, cond) do { \
	for (var = start; *var && (cond); var++) \
	    /* void */; \
    } while (0)

#define TRIM(s) do { \
	char *p; \
	for (p = (s) + strlen(s); p > (s) && ISSPACE(p[-1]); p--) \
	    /* void */; \
	*p = 0; \
    } while (0)

    SKIP(buf, np, ISSPACE(*np));		/* find name begin */
    if (*np == 0)
	return ("missing attribute name");
    SKIP(np, ep, !ISSPACE(*ep) && *ep != '=');	/* find name end */
    SKIP(ep, cp, ISSPACE(*cp));			/* skip blanks before '=' */
    if (*cp != '=')				/* need '=' */
	return ("missing '=' after attribute name");
    *ep = 0;					/* terminate name */
    cp++;					/* skip over '=' */
    SKIP(cp, vp, ISSPACE(*vp));			/* skip leading blanks */
    TRIM(vp);					/* trim trailing blanks */
    *name = np;
    *value = vp;
    return (0);
}
