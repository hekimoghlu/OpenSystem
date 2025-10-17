/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <mac_parse.h>

 /*
  * Helper macro for consistency. Null-terminate the temporary buffer,
  * execute the action, and reset the temporary buffer for re-use.
  */
#define MAC_PARSE_ACTION(status, type, buf, context) \
	do { \
	    VSTRING_TERMINATE(buf); \
	    status |= action((type), (buf), (context)); \
	    VSTRING_RESET(buf); \
	} while(0)

/* mac_parse - split string into literal text and macro references */

int     mac_parse(const char *value, MAC_PARSE_FN action, void *context)
{
    const char *myname = "mac_parse";
    VSTRING *buf = vstring_alloc(1);	/* result buffer */
    const char *vp;			/* value pointer */
    const char *pp;			/* open_paren pointer */
    const char *ep;			/* string end pointer */
    static char open_paren[] = "({";
    static char close_paren[] = ")}";
    int     level;
    int     status = 0;

#define SKIP(start, var, cond) do { \
        for (var = start; *var && (cond); var++) \
	    /* void */; \
    } while (0)

    if (msg_verbose > 1)
	msg_info("%s: %s", myname, value);

    for (vp = value; *vp;) {
	if (*vp != '$') {			/* ordinary character */
	    VSTRING_ADDCH(buf, *vp);
	    vp += 1;
	} else if (vp[1] == '$') {		/* $$ becomes $ */
	    VSTRING_ADDCH(buf, *vp);
	    vp += 2;
	} else {				/* found bare $ */
	    if (VSTRING_LEN(buf) > 0)
		MAC_PARSE_ACTION(status, MAC_PARSE_LITERAL, buf, context);
	    vp += 1;
	    pp = open_paren;
	    if (*vp == *pp || *vp == *++pp) {	/* ${x} or $(x) */
		level = 1;
		vp += 1;
		for (ep = vp; level > 0; ep++) {
		    if (*ep == 0) {
			msg_warn("truncated macro reference: \"%s\"", value);
			status |= MAC_PARSE_ERROR;
			break;
		    }
		    if (*ep == *pp)
			level++;
		    if (*ep == close_paren[pp - open_paren])
			level--;
		}
		if (status & MAC_PARSE_ERROR)
		    break;
		vstring_strncat(buf, vp, level > 0 ? ep - vp : ep - vp - 1);
		vp = ep;
	    } else {				/* plain $x */
		SKIP(vp, ep, ISALNUM(*ep) || *ep == '_');
		vstring_strncat(buf, vp, ep - vp);
		vp = ep;
	    }
	    if (VSTRING_LEN(buf) == 0) {
		status |= MAC_PARSE_ERROR;
		msg_warn("empty macro name: \"%s\"", value);
		break;
	    }
	    MAC_PARSE_ACTION(status, MAC_PARSE_EXPR, buf, context);
	}
    }
    if (VSTRING_LEN(buf) > 0 && (status & MAC_PARSE_ERROR) == 0)
	MAC_PARSE_ACTION(status, MAC_PARSE_LITERAL, buf, context);

    /*
     * Cleanup.
     */
    vstring_free(buf);

    return (status);
}

#ifdef TEST

 /*
  * Proof-of-concept test program. Read strings from stdin, print parsed
  * result to stdout.
  */
#include <vstring_vstream.h>

/* mac_parse_print - print parse tree */

static int mac_parse_print(int type, VSTRING *buf, void *unused_context)
{
    char   *type_name;

    switch (type) {
    case MAC_PARSE_EXPR:
	type_name = "MAC_PARSE_EXPR";
	break;
    case MAC_PARSE_LITERAL:
	type_name = "MAC_PARSE_LITERAL";
	break;
    default:
	msg_panic("unknown token type %d", type);
    }
    vstream_printf("%s \"%s\"\n", type_name, vstring_str(buf));
    return (0);
}

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *buf = vstring_alloc(1);

    while (vstring_fgets_nonl(buf, VSTREAM_IN)) {
	mac_parse(vstring_str(buf), mac_parse_print, (void *) 0);
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(buf);
    return (0);
}

#endif
