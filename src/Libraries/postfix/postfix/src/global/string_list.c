/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

/* Utility library. */

#include <match_list.h>

/* Global library. */

#include "string_list.h"

#ifdef TEST

#include <stdlib.h>
#include <unistd.h>
#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <msg_vstream.h>
#include <dict.h>
#include <stringops.h>			/* util_utf8_enable */

static void usage(char *progname)
{
    msg_fatal("usage: %s [-v] patterns string", progname);
}

int     main(int argc, char **argv)
{
    STRING_LIST *list;
    char   *string;
    int     ch;

    msg_vstream_init(argv[0], VSTREAM_ERR);

    while ((ch = GETOPT(argc, argv, "v")) > 0) {
	switch (ch) {
	case 'v':
	    msg_verbose++;
	    break;
	default:
	    usage(argv[0]);
	}
    }
    if (argc != optind + 2)
	usage(argv[0]);
    dict_allow_surrogate = 1;
    util_utf8_enable = 1;
    list = string_list_init("command line", MATCH_FLAG_RETURN, argv[optind]);
    string = argv[optind + 1];
    vstream_printf("%s: %s\n", string, string_list_match(list, string) ?
		   "YES" : list->error == 0 ? "NO" : "ERROR");
    vstream_fflush(VSTREAM_OUT);
    string_list_free(list);
    return (0);
}

#endif
