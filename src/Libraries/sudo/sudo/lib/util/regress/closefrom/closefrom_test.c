/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Test that sudo_closefrom() works as expected.
 */

int
main(int argc, char *argv[])
{
    int ch, fds[2], flag, maxfd, minfd, errors = 0, ntests = 0;
    initprogname(argc > 0 ? argv[0] : "closefrom_test");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    /* We use pipe() because it doesn't rely on the filesystem. */
    ntests++;
    if (pipe(fds) == -1) {
	sudo_warn("%s", "pipe");
	errors++;
	goto done;
    }
    maxfd = MAX(fds[0], fds[1]);
    minfd = MIN(fds[0], fds[1]);

    /* Close any fds greater than fds[0] and fds[1]. */
    sudo_closefrom(maxfd + 1);

    /* Verify that sudo_closefrom() didn't close fds[0] or fds[1]. */
    ntests++;
    if (fcntl(fds[0], F_GETFL, 0) == -1) {
	sudo_warnx("fd %d closed prematurely", fds[0]);
	errors++;
	goto done;
    }
    ntests++;
    if (fcntl(fds[1], F_GETFL, 0) == -1) {
	sudo_warnx("fd %d closed prematurely", fds[1]);
	errors++;
	goto done;
    }

    /* Close fds[0], fds[1] and above. */
    sudo_closefrom(minfd);

    /* Verify that sudo_closefrom() closed both fds. */
    ntests++;
    flag = fcntl(fds[0], F_GETFD, 0);
#ifdef __APPLE__
    /* We only set the close-on-exec flag on macOS. */
    if (flag == 1)
	flag = -1;
#endif
    if (flag != -1) {
	sudo_warnx("fd %d still open", fds[0]);
	errors++;
	goto done;
    }
    ntests++;
    flag = fcntl(fds[1], F_GETFD, 0);
#ifdef __APPLE__
    /* We only set the close-on-exec flag on macOS. */
    if (flag == 1)
	flag = -1;
#endif
    if (flag != -1) {
	sudo_warnx("fd %d still open", fds[1]);
	errors++;
	goto done;
    }

done:
    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    return errors;
}
