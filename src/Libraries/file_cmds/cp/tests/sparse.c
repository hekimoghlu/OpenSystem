/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sysexits.h>
#include <unistd.h>

static bool verbose;

/*
 * Returns true if the file named by its argument is sparse, i.e. if
 * seeking to SEEK_HOLE returns a different value than seeking to
 * SEEK_END.
 */
static bool
sparse(const char *filename)
{
	off_t hole, end;
	int fd;

	if ((fd = open(filename, O_RDONLY)) < 0 ||
	    (hole = lseek(fd, 0, SEEK_HOLE)) < 0 ||
	    (end = lseek(fd, 0, SEEK_END)) < 0)
		err(1, "%s", filename);
	close(fd);
	if (end > hole) {
		if (verbose)
			printf("%s: hole at %zu\n", filename, (size_t)hole);
		return (true);
	}
	return (false);
}

static void
usage(void)
{

	fprintf(stderr, "usage: sparse [-v] file [...]\n");
	exit(EX_USAGE);
}

int
main(int argc, char *argv[])
{
	int opt, rv;

	while ((opt = getopt(argc, argv, "v")) != -1) {
		switch (opt) {
		case 'v':
			verbose = true;
			break;
		default:
			usage();
			break;
		}
	}
	argc -= optind;
	argv += optind;
	if (argc == 0)
		usage();
	rv = EXIT_SUCCESS;
	while (argc-- > 0)
		if (!sparse(*argv++))
			rv = EXIT_FAILURE;
	exit(rv);
}
