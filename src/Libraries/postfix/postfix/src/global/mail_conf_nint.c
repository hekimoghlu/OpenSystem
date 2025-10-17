/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

/* convert_mail_conf_nint - look up and convert integer parameter value */

static int convert_mail_conf_nint(const char *name, int *intval)
{
    const char *strval;
    char   *end;
    long    longval;

    if ((strval = mail_conf_lookup_eval(name)) != 0) {
	errno = 0;
	*intval = longval = strtol(strval, &end, 10);
	if (*strval == 0 || *end != 0 || errno == ERANGE || longval != *intval)
	    msg_fatal("bad numerical configuration: %s = %s", name, strval);
	return (1);
    }
    return (0);
}

/* check_mail_conf_nint - validate integer value */

static void check_mail_conf_nint(const char *name, int intval, int min, int max)
{
    if (min && intval < min)
	msg_fatal("invalid %s parameter value %d < %d", name, intval, min);
    if (max && intval > max)
	msg_fatal("invalid %s parameter value %d > %d", name, intval, max);
}

/* get_mail_conf_nint - evaluate integer-valued configuration variable */

int     get_mail_conf_nint(const char *name, const char *defval, int min, int max)
{
    int     intval;

    if (convert_mail_conf_nint(name, &intval) == 0)
	set_mail_conf_nint(name, defval);
    if (convert_mail_conf_nint(name, &intval) == 0)
	msg_panic("get_mail_conf_nint: parameter not found: %s", name);
    check_mail_conf_nint(name, intval, min, max);
    return (intval);
}

/* get_mail_conf_nint2 - evaluate integer-valued configuration variable */

int     get_mail_conf_nint2(const char *name1, const char *name2, int defval,
			            int min, int max)
{
    int     intval;
    char   *name;

    name = concatenate(name1, name2, (char *) 0);
    if (convert_mail_conf_nint(name, &intval) == 0)
	set_mail_conf_nint_int(name, defval);
    if (convert_mail_conf_nint(name, &intval) == 0)
	msg_panic("get_mail_conf_nint2: parameter not found: %s", name);
    check_mail_conf_nint(name, intval, min, max);
    myfree(name);
    return (intval);
}

/* get_mail_conf_nint_fn - evaluate integer-valued configuration variable */

typedef const char *(*stupid_indent_int) (void);

int     get_mail_conf_nint_fn(const char *name, stupid_indent_int defval,
			              int min, int max)
{
    int     intval;

    if (convert_mail_conf_nint(name, &intval) == 0)
	set_mail_conf_nint(name, defval());
    if (convert_mail_conf_nint(name, &intval) == 0)
	msg_panic("get_mail_conf_nint_fn: parameter not found: %s", name);
    check_mail_conf_nint(name, intval, min, max);
    return (intval);
}

/* set_mail_conf_nint - update integer-valued configuration dictionary entry */

void    set_mail_conf_nint(const char *name, const char *value)
{
    mail_conf_update(name, value);
}

/* set_mail_conf_nint_int - update integer-valued configuration dictionary entry */

void    set_mail_conf_nint_int(const char *name, int value)
{
    char    buf[BUFSIZ];		/* yeah! crappy code! */

    sprintf(buf, "%d", value);			/* yeah! more crappy code! */
    mail_conf_update(name, buf);
}

/* get_mail_conf_nint_table - look up table of integers */

void    get_mail_conf_nint_table(const CONFIG_NINT_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_nint(table->name, table->defval,
					      table->min, table->max);
	table++;
    }
}

/* get_mail_conf_nint_fn_table - look up integers, defaults are functions */

void    get_mail_conf_nint_fn_table(const CONFIG_NINT_FN_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_nint_fn(table->name, table->defval,
						 table->min, table->max);
	table++;
    }
}
