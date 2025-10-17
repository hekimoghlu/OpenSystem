/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#include "namadr_list.h"

#ifdef TEST

#include <stdlib.h>
#include <unistd.h>
#include <msg.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <dict.h>
#include <stringops.h>			/* util_utf8_enable */

static void usage(char *progname)
{
    msg_fatal("usage: %s [-v] pattern_list hostname address", progname);
}

int     main(int argc, char **argv)
{
    NAMADR_LIST *list;
    char   *host;
    char   *addr;
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
    if (argc != optind + 3)
	usage(argv[0]);
    dict_allow_surrogate = 1;
    util_utf8_enable = 1;
    list = namadr_list_init("command line", MATCH_FLAG_PARENT
			    | MATCH_FLAG_RETURN, argv[optind]);
    host = argv[optind + 1];
    addr = argv[optind + 2];
    vstream_printf("%s/%s: %s\n", host, addr,
		   namadr_list_match(list, host, addr) ?
		   "YES" : list->error == 0 ? "NO" : "ERROR");
    vstream_fflush(VSTREAM_OUT);
    namadr_list_free(list);
    return (0);
}

#endif
