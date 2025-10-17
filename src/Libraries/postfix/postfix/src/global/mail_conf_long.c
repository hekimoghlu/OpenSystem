/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include <stdlib.h>
#include <stdio.h>			/* BUFSIZ */
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <dict.h>
#include <stringops.h>

/* Global library. */

#include "mail_conf.h"

/* convert_mail_conf_long - look up and convert integer parameter value */

static int convert_mail_conf_long(const char *name, long *longval)
{
    const char *strval;
    char   *end;

    if ((strval = mail_conf_lookup_eval(name)) != 0) {
	errno = 0;
	*longval = strtol(strval, &end, 10);
	if (*strval == 0 || *end != 0 || errno == ERANGE)
	    msg_fatal("bad numerical configuration: %s = %s", name, strval);
	return (1);
    }
    return (0);
}

/* check_mail_conf_long - validate integer value */

static void check_mail_conf_long(const char *name, long longval, long min, long max)
{
    if (min && longval < min)
	msg_fatal("invalid %s parameter value %ld < %ld", name, longval, min);
    if (max && longval > max)
	msg_fatal("invalid %s parameter value %ld > %ld", name, longval, max);
}

/* get_mail_conf_long - evaluate integer-valued configuration variable */

long    get_mail_conf_long(const char *name, long defval, long min, long max)
{
    long    longval;

    if (convert_mail_conf_long(name, &longval) == 0)
	set_mail_conf_long(name, longval = defval);
    check_mail_conf_long(name, longval, min, max);
    return (longval);
}

/* get_mail_conf_long2 - evaluate integer-valued configuration variable */

long    get_mail_conf_long2(const char *name1, const char *name2, long defval,
			            long min, long max)
{
    long    longval;
    char   *name;

    name = concatenate(name1, name2, (char *) 0);
    if (convert_mail_conf_long(name, &longval) == 0)
	set_mail_conf_long(name, longval = defval);
    check_mail_conf_long(name, longval, min, max);
    myfree(name);
    return (longval);
}

/* get_mail_conf_long_fn - evaluate integer-valued configuration variable */

typedef long (*stupid_indent_long) (void);

long    get_mail_conf_long_fn(const char *name, stupid_indent_long defval,
			              long min, long max)
{
    long    longval;

    if (convert_mail_conf_long(name, &longval) == 0)
	set_mail_conf_long(name, longval = defval());
    check_mail_conf_long(name, longval, min, max);
    return (longval);
}

/* set_mail_conf_long - update integer-valued configuration dictionary entry */

void    set_mail_conf_long(const char *name, long value)
{
    char    buf[BUFSIZ];		/* yeah! crappy code! */

    sprintf(buf, "%ld", value);			/* yeah! more crappy code! */
    mail_conf_update(name, buf);
}

/* get_mail_conf_long_table - look up table of integers */

void    get_mail_conf_long_table(const CONFIG_LONG_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_long(table->name, table->defval,
					      table->min, table->max);
	table++;
    }
}

/* get_mail_conf_long_fn_table - look up integers, defaults are functions */

void    get_mail_conf_long_fn_table(const CONFIG_LONG_FN_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_long_fn(table->name, table->defval,
						 table->min, table->max);
	table++;
    }
}
