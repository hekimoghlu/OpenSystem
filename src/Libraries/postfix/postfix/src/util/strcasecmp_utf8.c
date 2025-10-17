/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

 /*
  * Utility library.
  */
#include <stringops.h>

#define STR(x)	vstring_str(x)

static VSTRING *f1;			/* casefold result for s1 */
static VSTRING *f2;			/* casefold result for s2 */

/* strcasecmp_utf8_init - initialize */

static void strcasecmp_utf8_init(void)
{
    f1 = vstring_alloc(100);
    f2 = vstring_alloc(100);
}

/* strcasecmp_utf8x - caseless string comparison */

int     strcasecmp_utf8x(int flags, const char *s1, const char *s2)
{

    /*
     * Short-circuit optimization for ASCII-only text. This may be slower
     * than using a cache for all results. We must not expose strcasecmp(3)
     * to non-ASCII text.
     */
    if (allascii(s1) && allascii(s2))
	return (strcasecmp(s1, s2));

    if (f1 == 0)
	strcasecmp_utf8_init();

    /*
     * Cross our fingers and hope that strcmp() remains agnostic of
     * charactersets and locales.
     */
    flags &= CASEF_FLAG_UTF8;
    casefoldx(flags, f1, s1, -1);
    casefoldx(flags, f2, s2, -1);
    return (strcmp(STR(f1), STR(f2)));
}

/* strncasecmp_utf8x - caseless string comparison */

int     strncasecmp_utf8x(int flags, const char *s1, const char *s2,
			          ssize_t len)
{

    /*
     * Consider using a cache for all results.
     */
    if (f1 == 0)
	strcasecmp_utf8_init();

    /*
     * Short-circuit optimization for ASCII-only text. This may be slower
     * than using a cache for all results. See comments above for limitations
     * of strcasecmp().
     */
    if (allascii_len(s1, len) && allascii_len(s2, len))
	return (strncasecmp(s1, s2, len));

    /*
     * Caution: casefolding may change the number of bytes. See comments
     * above for concerns about strcmp().
     */
    flags &= CASEF_FLAG_UTF8;
    casefoldx(flags, f1, s1, len);
    casefoldx(flags, f2, s2, len);
    return (strcmp(STR(f1), STR(f2)));
}

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <msg_vstream.h>
#include <argv.h>

int     main(int argc, char **argv)
{
    VSTRING *buffer = vstring_alloc(1);
    ARGV   *cmd;
    char  **args;
    int     len;
    int     flags;
    int     res;

    msg_vstream_init(argv[0], VSTREAM_ERR);
    flags = CASEF_FLAG_UTF8;
    util_utf8_enable = 1;
    while (vstring_fgets_nonl(buffer, VSTREAM_IN)) {
	vstream_printf("> %s\n", STR(buffer));
	cmd = argv_split(STR(buffer), CHARS_SPACE);
	if (cmd->argc == 0 || cmd->argv[0][0] == '#')
	    continue;
	args = cmd->argv;

	/*
	 * Compare two strings.
	 */
	if (strcmp(args[0], "compare") == 0 && cmd->argc == 3) {
	    res = strcasecmp_utf8x(flags, args[1], args[2]);
	    vstream_printf("\"%s\" %s \"%s\"\n",
			   args[1],
			   res < 0 ? "<" : res == 0 ? "==" : ">",
			   args[2]);
	}

	/*
	 * Compare two substrings.
	 */
	else if (strcmp(args[0], "compare-len") == 0 && cmd->argc == 4
		 && sscanf(args[3], "%d", &len) == 1 && len >= 0) {
	    res = strncasecmp_utf8x(flags, args[1], args[2], len);
	    vstream_printf("\"%.*s\" %s \"%.*s\"\n",
			   len, args[1],
			   res < 0 ? "<" : res == 0 ? "==" : ">",
			   len, args[2]);
	}

	/*
	 * Usage.
	 */
	else {
	    vstream_printf("Usage: %s compare <s1> <s2> | compare-len <s1> <s2> <len>\n",
			   argv[0]);
	}
	vstream_fflush(VSTREAM_OUT);
	argv_free(cmd);
    }
    exit(0);
}

#endif					/* TEST */
