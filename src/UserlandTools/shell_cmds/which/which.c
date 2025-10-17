/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include <sys/cdefs.h>

__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/stat.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void	 usage(void);
static int	 print_matches(char *, char *);

static int 	 silent;
static int 	 allpaths;

int
main(int argc, char **argv)
{
	char *p, *path;
	ssize_t pathlen;
	int opt, status;

	status = EXIT_SUCCESS;

	while ((opt = getopt(argc, argv, "as")) != -1) {
		switch (opt) {
		case 'a':
			allpaths = 1;
			break;
		case 's':
			silent = 1;
			break;
		default:
			usage();
			break;
		}
	}

	argv += optind;
	argc -= optind;

	if (argc == 0)
		usage();

	if ((p = getenv("PATH")) == NULL)
		exit(EXIT_FAILURE);
	pathlen = strlen(p) + 1;
	path = malloc(pathlen);
	if (path == NULL)
		err(EXIT_FAILURE, NULL);

	while (argc > 0) {
		memcpy(path, p, pathlen);

		if (strlen(*argv) >= FILENAME_MAX ||
		    print_matches(path, *argv) == -1)
			status = EXIT_FAILURE;

		argv++;
		argc--;
	}

	exit(status);
}

static void
usage(void)
{

	(void)fprintf(stderr, "usage: which [-as] program ...\n");
	exit(EXIT_FAILURE);
}

static int
is_there(char *candidate)
{
	struct stat fin;

	/* XXX work around access(2) false positives for superuser */
	if (access(candidate, X_OK) == 0 &&
	    stat(candidate, &fin) == 0 &&
	    S_ISREG(fin.st_mode) &&
	    (getuid() != 0 ||
	    (fin.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) != 0)) {
		if (!silent)
			printf("%s\n", candidate);
		return (1);
	}
	return (0);
}

static int
print_matches(char *path, char *filename)
{
	char candidate[PATH_MAX];
	const char *d;
	int found;

	if (strchr(filename, '/') != NULL)
		return (is_there(filename) ? 0 : -1);
	found = 0;
	while ((d = strsep(&path, ":")) != NULL) {
		if (*d == '\0')
			d = ".";
		if (snprintf(candidate, sizeof(candidate), "%s/%s", d,
		    filename) >= (int)sizeof(candidate))
			continue;
		if (is_there(candidate)) {
			found = 1;
			if (!allpaths)
				break;
		}
	}
	return (found ? 0 : -1);
}

