/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <err.h>
#include <sysexits.h>

#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/param.h>
#include <sys/time.h>

#include <copyfile.h>

void usage(void);

int
main(int argc, char * argv[])
{
	struct stat sb;
	void *mset;
	mode_t mode;
	bool gotmode = false;
	int ch;
	int ret;
	int srcfd, dstfd;
	const char *src = NULL;
	const char *dst = NULL;
	char dsttmpname[MAXPATHLEN];

	while ((ch = getopt(argc, argv, "cSm:")) != -1) {
		switch (ch) {
		case 'c':
		case 'S':
			/* ignored for compatibility */
			break;
		case 'm':
			gotmode = true;
			mset = setmode(optarg);
			if (!mset) {
				errx(EX_USAGE, "Unrecognized mode %s", optarg);
			}

			mode = getmode(mset, 0);
			free(mset);
			break;
		case '?':
		default:
			usage();
		}
	}

	argc -= optind;
	argv += optind;

	if (argc < 2) {
		usage();
	}

	src = argv[0];
	dst = argv[1];

	srcfd = open(src, O_RDONLY, 0);
	if (srcfd < 0) {
		err(EX_NOINPUT, "open(%s)", src);
	}

	ret = fstat(srcfd, &sb);
	if (ret < 0) {
		err(EX_NOINPUT, "fstat(%s)", src);
	}

	if (!S_ISREG(sb.st_mode)) {
		err(EX_USAGE, "%s is not a regular file", src);
	}

	snprintf(dsttmpname, sizeof(dsttmpname), "%s.XXXXXX", dst);

	dstfd = mkstemp(dsttmpname);
	if (dstfd < 0) {
		err(EX_UNAVAILABLE, "mkstemp(%s)", dsttmpname);
	}

	ret = fcopyfile(srcfd, dstfd, NULL,
	    COPYFILE_DATA);
	if (ret < 0) {
		err(EX_UNAVAILABLE, "fcopyfile(%s, %s)", src, dsttmpname);
	}

	ret = futimes(dstfd, NULL);
	if (ret < 0) {
		err(EX_UNAVAILABLE, "futimes(%s)", dsttmpname);
	}

	if (gotmode) {
		ret = fchmod(dstfd, mode);
		if (ret < 0) {
			err(EX_NOINPUT, "fchmod(%s, %ho)", dsttmpname, mode);
		}
	}

	ret = rename(dsttmpname, dst);
	if (ret < 0) {
		err(EX_NOINPUT, "rename(%s, %s)", dsttmpname, dst);
	}

	ret = close(dstfd);
	if (ret < 0) {
		err(EX_NOINPUT, "close(dst)");
	}

	ret = close(srcfd);
	if (ret < 0) {
		err(EX_NOINPUT, "close(src)");
	}

	return 0;
}

void
usage(void)
{
	fprintf(stderr, "Usage: %s [-c] [-S] [-m <mode>] <src> <dst>\n",
	    getprogname());
	exit(EX_USAGE);
}
