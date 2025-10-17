/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <dict.h>
#include <stringops.h>

/* Global library. */

#include "conv_time.h"
#include "mail_conf.h"

/* convert_mail_conf_time - look up and convert integer parameter value */

static int convert_mail_conf_time(const char *name, int *intval, int def_unit)
{
    const char *strval;

    if ((strval = mail_conf_lookup_eval(name)) == 0)
	return (0);
    if (conv_time(strval, intval, def_unit) == 0)
	msg_fatal("parameter %s: bad time value or unit: %s", name, strval);
    return (1);
}

/* check_mail_conf_time - validate integer value */

void    check_mail_conf_time(const char *name, int intval, int min, int max)
{
    if (min && intval < min)
	msg_fatal("invalid %s: %d (min %d)", name, intval, min);
    if (max && intval > max)
	msg_fatal("invalid %s: %d (max %d)", name, intval, max);
}

/* get_def_time_unit - extract time unit from default value */

static int get_def_time_unit(const char *name, const char *defval)
{
    const char *cp;

    for (cp = mail_conf_eval(defval); /* void */ ; cp++) {
	if (*cp == 0)
	    msg_panic("parameter %s: missing time unit in default value: %s",
		      name, defval);
	if (ISALPHA(*cp)) {
#if 0
	    if (cp[1] != 0)
		msg_panic("parameter %s: bad time unit in default value: %s",
			  name, defval);
#endif
	    return (*cp);
	}
    }
}

/* get_mail_conf_time - evaluate integer-valued configuration variable */

int     get_mail_conf_time(const char *name, const char *defval, int min, int max)
{
    int     intval;
    int     def_unit;

    def_unit = get_def_time_unit(name, defval);
    if (convert_mail_conf_time(name, &intval, def_unit) == 0)
	set_mail_conf_time(name, defval);
    if (convert_mail_conf_time(name, &intval, def_unit) == 0)
	msg_panic("get_mail_conf_time: parameter not found: %s", name);
    check_mail_conf_time(name, intval, min, max);
    return (intval);
}

/* get_mail_conf_time2 - evaluate integer-valued configuration variable */

int     get_mail_conf_time2(const char *name1, const char *name2,
			         int defval, int def_unit, int min, int max)
{
    int     intval;
    char   *name;

    name = concatenate(name1, name2, (char *) 0);
    if (convert_mail_conf_time(name, &intval, def_unit) == 0)
	set_mail_conf_time_int(name, defval);
    if (convert_mail_conf_time(name, &intval, def_unit) == 0)
	msg_panic("get_mail_conf_time2: parameter not found: %s", name);
    check_mail_conf_time(name, intval, min, max);
    myfree(name);
    return (intval);
}

/* set_mail_conf_time - update integer-valued configuration dictionary entry */

void    set_mail_conf_time(const char *name, const char *value)
{
    mail_conf_update(name, value);
}

/* set_mail_conf_time_int - update integer-valued configuration dictionary entry */

void    set_mail_conf_time_int(const char *name, int value)
{
    char    buf[BUFSIZ];		/* yeah! crappy code! */

    sprintf(buf, "%ds", value);			/* yeah! more crappy code! */
    mail_conf_update(name, buf);
}

/* get_mail_conf_time_table - look up table of integers */

void    get_mail_conf_time_table(const CONFIG_TIME_TABLE *table)
{
    while (table->name) {
	table->target[0] = get_mail_conf_time(table->name, table->defval,
					      table->min, table->max);
	table++;
    }
}

#ifdef TEST

 /*
  * Stand-alone driver program for regression testing.
  */
#include <vstream.h>

int     main(int unused_argc, char **unused_argv)
{
    static int seconds;
    static int minutes;
    static int hours;
    static int days;
    static int weeks;
    static const CONFIG_TIME_TABLE time_table[] = {
	"seconds", "10s", &seconds, 0, 0,
	"minutes", "10m", &minutes, 0, 0,
	"hours", "10h", &hours, 0, 0,
	"days", "10d", &days, 0, 0,
	"weeks", "10w", &weeks, 0, 0,
	0,
    };

    get_mail_conf_time_table(time_table);
    vstream_printf("10 seconds = %d\n", seconds);
    vstream_printf("10 minutes = %d\n", minutes);
    vstream_printf("10 hours = %d\n", hours);
    vstream_printf("10 days = %d\n", days);
    vstream_printf("10 weeks = %d\n", weeks);
    vstream_fflush(VSTREAM_OUT);
    return (0);
}

#endif
