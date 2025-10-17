/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <dict.h>

/* Global library. */

#include "mail_conf.h"

/* convert_mail_conf_nbool - look up and convert boolean parameter value */

static int convert_mail_conf_nbool(const char *name, int *intval)
{
    const char *strval;

    if ((strval = mail_conf_lookup_eval(name)) == 0) {
	return (0);
    } else {
	if (strcasecmp(strval, CONFIG_BOOL_YES) == 0) {
	    *intval = 1;
	} else if (strcasecmp(strval, CONFIG_BOOL_NO) == 0) {
	    *intval = 0;
	} else {
	    msg_fatal("bad boolean configuration: %s = %s", name, strval);
	}
	return (1);
    }
}

/* get_mail_conf_nbool - evaluate boolean-valued configuration variable */

int     get_mail_conf_nbool(const char *name, const char *defval)
{
    int     intval;

    if (convert_mail_conf_nbool(name, &intval) == 0)
	set_mail_conf_nbool(name, defval);
    if (convert_mail_conf_nbool(name, &intval) == 0)
	msg_panic("get_mail_conf_nbool: parameter not found: %s", name);
    return (intval);
}

/* get_mail_conf_nbool_fn - evaluate boolean-valued configuration variable */

typedef const char *(*stupid_indent_int) (void);

int     get_mail_conf_nbool_fn(const char *name, stupid_indent_int defval)
{
    int     intval;

    if (convert_mail_conf_nbool(name, &intval) == 0)
	set_mail_conf_nbool(name, defval());
    if (convert_mail_conf_nbool(name, &intval) == 0)
	msg_panic("get_mail_conf_nbool_fn: parameter not found: %s", name);
    return (intval);
}

/* set_mail_conf_nbool - update boolean-valued configuration dictionary entry */

void    set_mail_conf_nbool(const char *name, const char *value)
{
    mail_conf_update(name, value);
}

/* get_mail_conf_nbool_table - look up table of booleans */

void    get_mail_conf_nbool_table(const CONFIG_NBOOL_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_nbool(table->name, table->defval);
	table++;
    }
}

/* get_mail_conf_nbool_fn_table - look up booleans, defaults are functions */

void    get_mail_conf_nbool_fn_table(const CONFIG_NBOOL_FN_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_nbool_fn(table->name, table->defval);
	table++;
    }
}
