/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "includes.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_OPENS 10

void
fail(char *msg)
{
	fprintf(stderr, "closefrom: %s\n", msg);
	exit(1);
}

int
main(void)
{
	int i, max, fds[NUM_OPENS];
	char buf[512];

	for (i = 0; i < NUM_OPENS; i++)
		if ((fds[i] = open("/dev/null", O_RDONLY)) == -1)
			exit(0);	/* can't test */
	max = i - 1;

	/* should close last fd only */
	closefrom(fds[max]);
	if (close(fds[max]) != -1)
		fail("failed to close highest fd");

	/* make sure we can still use remaining descriptors */
	for (i = 0; i < max; i++)
		if (read(fds[i], buf, sizeof(buf)) == -1)
			fail("closed descriptors it should not have");

	/* should close all fds */
	closefrom(fds[0]);
	for (i = 0; i < NUM_OPENS; i++)
		if (close(fds[i]) != -1)
			fail("failed to close from lowest fd");
	return 0;
}
