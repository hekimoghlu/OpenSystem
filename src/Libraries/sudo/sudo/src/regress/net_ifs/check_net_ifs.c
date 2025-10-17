/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include <config.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

extern int get_net_ifs(char **addrinfo);

int
main(int argc, char *argv[])
{
    int ch, ninterfaces, errors = 0, ntests = 1;
    char *interfaces = NULL;
    bool verbose = false;

    initprogname(argc > 0 ? argv[0] : "check_net_ifs");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    verbose = true;
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }

    ninterfaces = get_net_ifs(&interfaces);
    switch (ninterfaces) {
    case -1:
	printf("FAIL: unable to get network interfaces\n");
	errors++;
	break;
    case 0:
	/* no interfaces or STUB_LOAD_INTERFACES defined. */
	if (verbose)
	    printf("OK: (0 interfaces)\n");
	break;
    default:
	if (verbose) {
	    printf("OK: (%d interface%s, %s)\n", ninterfaces,
		ninterfaces > 1 ? "s" : "", interfaces);
	}
	break;
    }
    free(interfaces);

    if (ntests != 0) {
        printf("%s: %d tests run, %d errors, %d%% success rate\n",
            getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    return errors;
}
