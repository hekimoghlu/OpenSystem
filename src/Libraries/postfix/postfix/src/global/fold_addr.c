/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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

#include <stringops.h>

/* Global library. */

#include <fold_addr.h>

#define STR(x)	vstring_str(x)

/* fold_addr - case fold mail address */

char   *fold_addr(VSTRING *result, const char *addr, int flags)
{
    char   *cp;

    /*
     * Fold the address as appropriate.
     */
    switch (flags & FOLD_ADDR_ALL) {
    case FOLD_ADDR_HOST:
	if ((cp = strrchr(addr, '@')) != 0) {
	    cp += 1;
	    vstring_strncpy(result, addr, cp - addr);
	    casefold_append(result, cp);
	    break;
	}
	/* FALLTHROUGH */
    case 0:
	vstring_strcpy(result, addr);
	break;
    case FOLD_ADDR_USER:
	if ((cp = strrchr(addr, '@')) != 0) {
	    casefold_len(result, addr, cp - addr);
	    vstring_strcat(result, cp);
	    break;
	}
	/* FALLTHROUGH */
    case FOLD_ADDR_USER | FOLD_ADDR_HOST:
	casefold(result, addr);
	break;
    }
    return (STR(result));
}

#ifdef TEST
#include <stdlib.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <msg_vstream.h>
#include <argv.h>

int     main(int argc, char **argv)
{
    VSTRING *line_buffer = vstring_alloc(1);
    VSTRING *fold_buffer = vstring_alloc(1);
    ARGV   *cmd;
    char  **args;

    msg_vstream_init(argv[0], VSTREAM_ERR);
    util_utf8_enable = 1;
    while (vstring_fgets_nonl(line_buffer, VSTREAM_IN)) {
	vstream_printf("> %s\n", STR(line_buffer));
	cmd = argv_split(STR(line_buffer), CHARS_SPACE);
	if (cmd->argc == 0 || cmd->argv[0][0] == '#') {
	    argv_free(cmd);
	    continue;
	}
	args = cmd->argv;

	/*
	 * Fold the host.
	 */
	if (strcmp(args[0], "host") == 0 && cmd->argc == 2) {
	    vstream_printf("\"%s\" -> \"%s\"\n", args[1], fold_addr(fold_buffer,
						  args[1], FOLD_ADDR_HOST));
	}

	/*
	 * Fold the user.
	 */
	else if (strcmp(args[0], "user") == 0 && cmd->argc == 2) {
	    vstream_printf("\"%s\" -> \"%s\"\n", args[1], fold_addr(fold_buffer,
						  args[1], FOLD_ADDR_USER));
	}

	/*
	 * Fold user and host.
	 */
	else if (strcmp(args[0], "all") == 0 && cmd->argc == 2) {
	    vstream_printf("\"%s\" -> \"%s\"\n", args[1], fold_addr(fold_buffer,
						   args[1], FOLD_ADDR_ALL));
	}

	/*
	 * Fold none.
	 */
	else if (strcmp(args[0], "none") == 0 && cmd->argc == 2) {
	    vstream_printf("\"%s\" -> \"%s\"\n", args[1], fold_addr(fold_buffer,
							       args[1], 0));
	}

	/*
	 * Usage.
	 */
	else {
	    vstream_printf("Usage: %s host <addr> | user <addr> | all <addr>\n",
			   argv[0]);
	}
	vstream_fflush(VSTREAM_OUT);
	argv_free(cmd);
    }
    vstring_free(line_buffer);
    vstring_free(fold_buffer);
    exit(0);
}

#endif					/* TEST */
