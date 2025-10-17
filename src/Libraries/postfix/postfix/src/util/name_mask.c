/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#include <errno.h>
#include <stdlib.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <name_mask.h>
#include <vstring.h>

static int hex_to_ulong(char *, unsigned long, unsigned long *);

#define STR(x) vstring_str(x)

/* name_mask_delim_opt - compute mask corresponding to list of names */

int     name_mask_delim_opt(const char *context, const NAME_MASK *table,
		            const char *names, const char *delim, int flags)
{
    const char *myname = "name_mask";
    char   *saved_names = mystrdup(names);
    char   *bp = saved_names;
    int     result = 0;
    const NAME_MASK *np;
    char   *name;
    int     (*lookup) (const char *, const char *);
    unsigned long ulval;

    if ((flags & NAME_MASK_REQUIRED) == 0)
	msg_panic("%s: missing NAME_MASK_FATAL/RETURN/WARN/IGNORE flag",
		  myname);

    if (flags & NAME_MASK_ANY_CASE)
	lookup = strcasecmp;
    else
	lookup = strcmp;

    /*
     * Break up the names string, and look up each component in the table. If
     * the name is found, merge its mask with the result.
     */
    while ((name = mystrtok(&bp, delim)) != 0) {
	for (np = table; /* void */ ; np++) {
	    if (np->name == 0) {
		if ((flags & NAME_MASK_NUMBER)
		    && hex_to_ulong(name, ~0U, &ulval)) {
		    result |= (unsigned int) ulval;
		} else if (flags & NAME_MASK_FATAL) {
		    msg_fatal("unknown %s value \"%s\" in \"%s\"",
			      context, name, names);
		} else if (flags & NAME_MASK_RETURN) {
		    msg_warn("unknown %s value \"%s\" in \"%s\"",
			     context, name, names);
		    myfree(saved_names);
		    return (0);
		} else if (flags & NAME_MASK_WARN) {
		    msg_warn("unknown %s value \"%s\" in \"%s\"",
			     context, name, names);
		}
		break;
	    }
	    if (lookup(name, np->name) == 0) {
		if (msg_verbose)
		    msg_info("%s: %s", myname, name);
		result |= np->mask;
		break;
	    }
	}
    }
    myfree(saved_names);
    return (result);
}

/* str_name_mask_opt - mask to string */

const char *str_name_mask_opt(VSTRING *buf, const char *context,
			              const NAME_MASK *table,
			              int mask, int flags)
{
    const char *myname = "name_mask";
    const NAME_MASK *np;
    ssize_t len;
    static VSTRING *my_buf = 0;
    int     delim = (flags & NAME_MASK_COMMA ? ',' :
		     (flags & NAME_MASK_PIPE ? '|' : ' '));

    if ((flags & STR_NAME_MASK_REQUIRED) == 0)
	msg_panic("%s: missing NAME_MASK_NUMBER/FATAL/RETURN/WARN/IGNORE flag",
		  myname);

    if (buf == 0) {
	if (my_buf == 0)
	    my_buf = vstring_alloc(1);
	buf = my_buf;
    }
    VSTRING_RESET(buf);

    for (np = table; mask != 0; np++) {
	if (np->name == 0) {
	    if (flags & NAME_MASK_NUMBER) {
		vstring_sprintf_append(buf, "0x%x%c", mask, delim);
	    } else if (flags & NAME_MASK_FATAL) {
		msg_fatal("%s: unknown %s bit in mask: 0x%x",
			  myname, context, mask);
	    } else if (flags & NAME_MASK_RETURN) {
		msg_warn("%s: unknown %s bit in mask: 0x%x",
			 myname, context, mask);
		return (0);
	    } else if (flags & NAME_MASK_WARN) {
		msg_warn("%s: unknown %s bit in mask: 0x%x",
			 myname, context, mask);
	    }
	    break;
	}
	if (mask & np->mask) {
	    mask &= ~np->mask;
	    vstring_sprintf_append(buf, "%s%c", np->name, delim);
	}
    }
    if ((len = VSTRING_LEN(buf)) > 0)
	vstring_truncate(buf, len - 1);
    VSTRING_TERMINATE(buf);

    return (STR(buf));
}

/* long_name_mask_delim_opt - compute mask corresponding to list of names */

long    long_name_mask_delim_opt(const char *context,
				         const LONG_NAME_MASK * table,
			               const char *names, const char *delim,
				         int flags)
{
    const char *myname = "name_mask";
    char   *saved_names = mystrdup(names);
    char   *bp = saved_names;
    long    result = 0;
    const LONG_NAME_MASK *np;
    char   *name;
    int     (*lookup) (const char *, const char *);
    unsigned long ulval;

    if ((flags & NAME_MASK_REQUIRED) == 0)
	msg_panic("%s: missing NAME_MASK_FATAL/RETURN/WARN/IGNORE flag",
		  myname);

    if (flags & NAME_MASK_ANY_CASE)
	lookup = strcasecmp;
    else
	lookup = strcmp;

    /*
     * Break up the names string, and look up each component in the table. If
     * the name is found, merge its mask with the result.
     */
    while ((name = mystrtok(&bp, delim)) != 0) {
	for (np = table; /* void */ ; np++) {
	    if (np->name == 0) {
		if ((flags & NAME_MASK_NUMBER)
		    && hex_to_ulong(name, ~0UL, &ulval)) {
		    result |= ulval;
		} else if (flags & NAME_MASK_FATAL) {
		    msg_fatal("unknown %s value \"%s\" in \"%s\"",
			      context, name, names);
		} else if (flags & NAME_MASK_RETURN) {
		    msg_warn("unknown %s value \"%s\" in \"%s\"",
			     context, name, names);
		    myfree(saved_names);
		    return (0);
		} else if (flags & NAME_MASK_WARN) {
		    msg_warn("unknown %s value \"%s\" in \"%s\"",
			     context, name, names);
		}
		break;
	    }
	    if (lookup(name, np->name) == 0) {
		if (msg_verbose)
		    msg_info("%s: %s", myname, name);
		result |= np->mask;
		break;
	    }
	}
    }

    myfree(saved_names);
    return (result);
}

/* str_long_name_mask_opt - mask to string */

const char *str_long_name_mask_opt(VSTRING *buf, const char *context,
				           const LONG_NAME_MASK * table,
				           long mask, int flags)
{
    const char *myname = "name_mask";
    ssize_t len;
    static VSTRING *my_buf = 0;
    int     delim = (flags & NAME_MASK_COMMA ? ',' :
		     (flags & NAME_MASK_PIPE ? '|' : ' '));
    const LONG_NAME_MASK *np;

    if ((flags & STR_NAME_MASK_REQUIRED) == 0)
	msg_panic("%s: missing NAME_MASK_NUMBER/FATAL/RETURN/WARN/IGNORE flag",
		  myname);

    if (buf == 0) {
	if (my_buf == 0)
	    my_buf = vstring_alloc(1);
	buf = my_buf;
    }
    VSTRING_RESET(buf);

    for (np = table; mask != 0; np++) {
	if (np->name == 0) {
	    if (flags & NAME_MASK_NUMBER) {
		vstring_sprintf_append(buf, "0x%lx%c", mask, delim);
	    } else if (flags & NAME_MASK_FATAL) {
		msg_fatal("%s: unknown %s bit in mask: 0x%lx",
			  myname, context, mask);
	    } else if (flags & NAME_MASK_RETURN) {
		msg_warn("%s: unknown %s bit in mask: 0x%lx",
			 myname, context, mask);
		return (0);
	    } else if (flags & NAME_MASK_WARN) {
		msg_warn("%s: unknown %s bit in mask: 0x%lx",
			 myname, context, mask);
	    }
	    break;
	}
	if (mask & np->mask) {
	    mask &= ~np->mask;
	    vstring_sprintf_append(buf, "%s%c", np->name, delim);
	}
    }
    if ((len = VSTRING_LEN(buf)) > 0)
	vstring_truncate(buf, len - 1);
    VSTRING_TERMINATE(buf);

    return (STR(buf));
}

/* hex_to_ulong - 0x... to unsigned long or smaller */

static int hex_to_ulong(char *value, unsigned long mask, unsigned long *ulp)
{
    unsigned long result;
    char   *cp;

    if (strncasecmp(value, "0x", 2) != 0)
	return (0);

    /*
     * Check for valid hex number. Since the value starts with 0x, strtoul()
     * will not allow a negative sign before the first nibble. So we don't
     * need to worry about explicit +/- signs.
     */
    errno = 0;
    result = strtoul(value, &cp, 16);
    if (*cp != '\0' || errno == ERANGE)
	return (0);

    *ulp = (result & mask);
    return (*ulp == result);
}

#ifdef TEST

 /*
  * Stand-alone test program.
  */
#include <stdlib.h>
#include <vstream.h>
#include <vstring_vstream.h>

int     main(int argc, char **argv)
{
    static const NAME_MASK demo_table[] = {
	"zero", 1 << 0,
	"one", 1 << 1,
	"two", 1 << 2,
	"three", 1 << 3,
	0, 0,
    };
    static const NAME_MASK feature_table[] = {
	"DEFAULT", NAME_MASK_DEFAULT,
	"FATAL", NAME_MASK_FATAL,
	"ANY_CASE", NAME_MASK_ANY_CASE,
	"RETURN", NAME_MASK_RETURN,
	"COMMA", NAME_MASK_COMMA,
	"PIPE", NAME_MASK_PIPE,
	"NUMBER", NAME_MASK_NUMBER,
	"WARN", NAME_MASK_WARN,
	"IGNORE", NAME_MASK_IGNORE,
	0,
    };
    int     in_feature_mask;
    int     out_feature_mask;
    int     demo_mask;
    const char *demo_str;
    VSTRING *out_buf = vstring_alloc(1);
    VSTRING *in_buf = vstring_alloc(1);

    if (argc != 3)
	msg_fatal("usage: %s in-feature-mask out-feature-mask", argv[0]);
    in_feature_mask = name_mask(argv[1], feature_table, argv[1]);
    out_feature_mask = name_mask(argv[2], feature_table, argv[2]);
    while (vstring_get_nonl(in_buf, VSTREAM_IN) != VSTREAM_EOF) {
	demo_mask = name_mask_opt("name", demo_table,
				  STR(in_buf), in_feature_mask);
	demo_str = str_name_mask_opt(out_buf, "mask", demo_table,
				     demo_mask, out_feature_mask);
	vstream_printf("%s -> 0x%x -> %s\n",
		       STR(in_buf), demo_mask,
		       demo_str ? demo_str : "(null)");
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(in_buf);
    vstring_free(out_buf);
    exit(0);
}

#endif
