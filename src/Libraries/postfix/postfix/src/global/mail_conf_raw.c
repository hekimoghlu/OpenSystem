/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>

/* Global library. */

#include "mail_conf.h"

/* check_mail_conf_raw - validate string length */

static void check_mail_conf_raw(const char *name, const char *strval,
			             int min, int max)
{
    ssize_t len = strlen(strval);

    if (min && len < min)
	msg_fatal("bad string length (%ld < %d): %s = %s",
		  (long) len, min, name, strval);
    if (max && len > max)
	msg_fatal("bad string length (%ld > %d): %s = %s",
		  (long) len, max, name, strval);
}

/* get_mail_conf_raw - evaluate string-valued configuration variable */

char   *get_mail_conf_raw(const char *name, const char *defval,
		               int min, int max)
{
    const char *strval;

    if ((strval = mail_conf_lookup(name)) == 0) {
	strval = defval;
	mail_conf_update(name, strval);
    }
    check_mail_conf_raw(name, strval, min, max);
    return (mystrdup(strval));
}

/* get_mail_conf_raw_fn - evaluate string-valued configuration variable */

typedef const char *(*stupid_indent_str) (void);

char   *get_mail_conf_raw_fn(const char *name, stupid_indent_str defval,
			          int min, int max)
{
    const char *strval;

    if ((strval = mail_conf_lookup(name)) == 0) {
	strval = defval();
	mail_conf_update(name, strval);
    }
    check_mail_conf_raw(name, strval, min, max);
    return (mystrdup(strval));
}

/* get_mail_conf_raw_table - look up table of strings */

void    get_mail_conf_raw_table(const CONFIG_RAW_TABLE *table)
{
    while (table->name) {
	if (table->target[0])
	    myfree(table->target[0]);
	table->target[0] = get_mail_conf_raw(table->name, table->defval,
					  table->min, table->max);
	table++;
    }
}

/* get_mail_conf_raw_fn_table - look up strings, defaults are functions */

void    get_mail_conf_raw_fn_table(const CONFIG_RAW_FN_TABLE *table)
{
    while (table->name) {
	if (table->target[0])
	    myfree(table->target[0]);
	table->target[0] = get_mail_conf_raw_fn(table->name, table->defval,
					     table->min, table->max);
	table++;
    }
}
