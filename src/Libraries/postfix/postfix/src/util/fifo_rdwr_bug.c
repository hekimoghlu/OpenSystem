/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include <sys_defs.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define FIFO_PATH       "test-fifo"
#define perrorexit(s)   { perror(s); exit(1); }

static void cleanup(void)
{
    printf("Removing fifo %s...\n", FIFO_PATH);
    if (unlink(FIFO_PATH))
	perrorexit("unlink");
    printf("Done.\n");
}

int     main(int unused_argc, char **unused_argv)
{
    struct timeval tv;
    fd_set  read_fds;
    fd_set  except_fds;
    int     fd;

    (void) unlink(FIFO_PATH);

    printf("Creating fifo %s...\n", FIFO_PATH);
    if (mkfifo(FIFO_PATH, 0600) < 0)
	perrorexit("mkfifo");

    printf("Opening fifo %s, read-write mode...\n", FIFO_PATH);
    if ((fd = open(FIFO_PATH, O_RDWR, 0)) < 0) {
	perror("open");
	cleanup();
	exit(1);
    }
    printf("Selecting the fifo for readability...\n");
    FD_ZERO(&read_fds);
    FD_SET(fd, &read_fds);
    FD_ZERO(&except_fds);
    FD_SET(fd, &except_fds);
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    switch (select(fd + 1, &read_fds, (fd_set *) 0, &except_fds, &tv)) {
    case -1:
	perrorexit("select");
    default:
	if (FD_ISSET(fd, &read_fds)) {
	    printf("Opening a fifo read-write makes it readable!!\n");
	    break;
	}
    case 0:
	printf("The fifo is not readable, as it should be.\n");
	break;
    }
    cleanup();
    exit(0);
}
