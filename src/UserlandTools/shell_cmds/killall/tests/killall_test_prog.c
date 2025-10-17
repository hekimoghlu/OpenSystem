/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int
main(int argc, char *argv[])
{

	if (argc == 1) {
		errx(1, "expected filename");
	} else if (strcmp(argv[1], "--child") != 0) {
		/*
		 * Pass on the argument, which should be the name of a flag
		 * file that the test will wait on, to the child to create.
		 */
		char * const nargv[] = { "innocent_test_prog",
		   "--child", argv[1], NULL };
		char * const envp[] = { NULL };

		execve(argv[0], nargv, envp);
		err(1, "execve");
	} else {
		int fd;

		argc -= 2;
		argv += 2;

		if (argc == 0)
			errx(1, "expected filename after --child");

		fd = open(argv[0], O_RDWR | O_CREAT, 0755);
		if (fd < 0)
			err(1, "%s", argv[0]);
		close(fd);

		while (1) {
			printf("Awaiting termination... ");
			sleep(1);
		}
	}

}
