/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

#define FIFO_PATH	"test-fifo"
#define perrorexit(s)	{ perror(s); exit(1); }

static void cleanup(void)
{
    printf("Removing fifo %s...\n", FIFO_PATH);
    if (unlink(FIFO_PATH))
	perrorexit("unlink");
    printf("Done.\n");
}

static void stuck(int unused_sig)
{
    printf("Non-blocking, write-only open of FIFO blocked\n");
    cleanup();
    exit(1);
}

int     main(int unused_argc, char **unused_argv)
{
    (void) unlink(FIFO_PATH);
    printf("Creating fifo %s...\n", FIFO_PATH);
    if (mkfifo(FIFO_PATH, 0600) < 0)
	perrorexit("mkfifo");
    signal(SIGALRM, stuck);
    alarm(5);
    printf("Opening fifo %s, non-blocking, write-only mode...\n", FIFO_PATH);
    if (open(FIFO_PATH, O_WRONLY | O_NONBLOCK, 0) < 0) {
	perror("open");
	cleanup();
	exit(1);
    }
    printf("Non-blocking, write-only open of FIFO succeeded\n");
    cleanup();
    exit(0);
}
