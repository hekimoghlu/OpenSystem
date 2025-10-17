/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>			/* strtol() */

 /*
  * Utility library.
  */
#include <msg.h>
#include <stringops.h>

 /*
  * Global library.
  */
#include <mail_conf.h>
#include <conv_time.h>
#include <attr_override.h>

/* attr_override - apply settings from list of attribute=value pairs */

void    attr_override(char *cp, const char *sep, const char *parens,...)
{
    static const char myname[] = "attr_override";
    va_list ap;
    int     idx;
    char   *nameval;
    const ATTR_OVER_INT *int_table = 0;
    const ATTR_OVER_STR *str_table = 0;
    const ATTR_OVER_TIME *time_table = 0;

    /*
     * Get the lookup tables and assignment targets.
     */
    va_start(ap, parens);
    while ((idx = va_arg(ap, int)) != ATTR_OVER_END) {
	switch (idx) {
	case ATTR_OVER_INT_TABLE:
	    if (int_table)
		msg_panic("%s: multiple ATTR_OVER_INT_TABLE", myname);
	    int_table = va_arg(ap, const ATTR_OVER_INT *);
	    break;
	case ATTR_OVER_STR_TABLE:
	    if (str_table)
		msg_panic("%s: multiple ATTR_OVER_STR_TABLE", myname);
	    str_table = va_arg(ap, const ATTR_OVER_STR *);
	    break;
	case ATTR_OVER_TIME_TABLE:
	    if (time_table)
		msg_panic("%s: multiple ATTR_OVER_TIME_TABLE", myname);
	    time_table = va_arg(ap, const ATTR_OVER_TIME *);
	    break;
	default:
	    msg_panic("%s: unknown argument type: %d", myname, idx);
	}
    }
    va_end(ap);

    /*
     * Process each attribute=value override in the input string.
     */
    while ((nameval = mystrtokq(&cp, sep, parens)) != 0) {
	int     found = 0;
	char   *key;
	char   *value;
	const char *err;
	const ATTR_OVER_INT *ip;
	const ATTR_OVER_STR *sp;
	const ATTR_OVER_TIME *tp;
	int     int_val;
	int     def_unit;
	char   *end;
	long    longval;

	/*
	 * Split into name and value.
	 */
	/* { name = value } */
	if (*nameval == parens[0]
	    && (err = extpar(&nameval, parens, EXTPAR_FLAG_NONE)) != 0)
	    msg_fatal("%s in \"%s\"", err, nameval);
	if ((err = split_nameval(nameval, &key, &value)) != 0)
	    msg_fatal("malformed option: %s: \"...%s...\"", err, nameval);

	/*
	 * Look up the name and apply the value.
	 */
	for (sp = str_table; sp != 0 && found == 0 && sp->name != 0; sp++) {
	    if (strcmp(sp->name, key) != 0)
		continue;
	    check_mail_conf_str(sp->name, value, sp->min, sp->max);
	    sp->target[0] = value;
	    found = 1;
	}
	for (ip = int_table; ip != 0 && found == 0 && ip->name != 0; ip++) {
	    if (strcmp(ip->name, key) != 0)
		continue;
	    /* XXX Duplicated from mail_conf_int(3). */
	    errno = 0;
	    int_val = longval = strtol(value, &end, 10);
	    if (*value == 0 || *end != 0 || errno == ERANGE
		|| longval != int_val)
		msg_fatal("bad numerical configuration: %s = %s", key, value);
	    check_mail_conf_int(key, int_val, ip->min, ip->max);
	    ip->target[0] = int_val;
	    found = 1;
	}
	for (tp = time_table; tp != 0 && found == 0 && tp->name != 0; tp++) {
	    if (strcmp(tp->name, key) != 0)
		continue;
	    def_unit = tp->defval[strspn(tp->defval, "0123456789")];
	    if (conv_time(value, &int_val, def_unit) == 0)
		msg_fatal("%s: bad time value or unit: %s", key, value);
	    check_mail_conf_time(key, int_val, tp->min, tp->max);
	    tp->target[0] = int_val;
	    found = 1;
	}
	if (found == 0)
	    msg_fatal("unknown option: \"%s = %s\"", key, value);
    }
}
