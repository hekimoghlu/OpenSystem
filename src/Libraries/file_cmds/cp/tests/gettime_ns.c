/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void
print_time(struct timespec *tv)
{

	printf("%ld", tv->tv_sec);
	if (tv->tv_nsec != 0)
		printf(".%ld", tv->tv_nsec);
	printf("\n");
}

int
main(int argc, char *argv[])
{
	struct stat sb;
	const char *file;

	if (argc != 2) {
		fprintf(stderr, "usage: %s file\n", getprogname());
		return (1);
	}

	file = argv[1];
	if (stat(file, &sb) != 0)
		err(1, "stat");

	/* atime, mtime */
	print_time(&sb.st_atimespec);
	print_time(&sb.st_mtimespec);
}
