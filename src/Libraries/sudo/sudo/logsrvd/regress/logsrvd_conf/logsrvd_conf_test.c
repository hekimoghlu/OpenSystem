/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#include <sys/socket.h>

#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_iolog.h"
#include "sudo_queue.h"
#include "logsrvd.h"

sudo_dso_public int main(int argc, char *argv[]);

static void
usage(void)
{
    fprintf(stderr, "usage: %s [-v] conf_file\n", getprogname());
    exit(EXIT_FAILURE);
}

/*
 * Simple test driver for logsrvd_conf_read().
 * Just pases the file, errors to standard error.
 */
int
main(int argc, char *argv[])
{
    bool verbose = false;
    int ch, ntests, errors = 0;

    initprogname(argc > 0 ? argv[0] : "conf_test");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    verbose = true;
	    break;
	default:
	    usage();
	}
    }
    argc -= optind;
    argv += optind;

    if (argc < 1)
	usage();

    for (ntests = 0; ntests < argc; ntests++) {
	const char *path = argv[ntests];
	if (verbose)
	    printf("reading %s\n", path);
	if (!logsrvd_conf_read(path))
	    errors++;
    }
    logsrvd_conf_cleanup();

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    return errors;
}
