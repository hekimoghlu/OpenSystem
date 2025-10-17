/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

/* Utility library. */

#include <name_code.h>

/* name_code - look up code by name */

int     name_code(const NAME_CODE *table, int flags, const char *name)
{
    const NAME_CODE *np;
    int     (*lookup) (const char *, const char *);

    if (flags & NAME_CODE_FLAG_STRICT_CASE)
	lookup = strcmp;
    else
	lookup = strcasecmp;

    for (np = table; np->name; np++)
	if (lookup(name, np->name) == 0)
	    break;
    return (np->code);
}

/* str_name_code - look up name by code */

const char *str_name_code(const NAME_CODE *table, int code)
{
    const NAME_CODE *np;

    for (np = table; np->name; np++)
	if (code == np->code)
	    break;
    return (np->name);
}
