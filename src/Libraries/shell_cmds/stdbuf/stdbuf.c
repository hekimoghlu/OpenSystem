/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef __APPLE__
#define	LIBSTDBUF	"/usr/lib/libstdbuf.so"
#define	LIBSTDBUF32	"/usr/lib32/libstdbuf.so"

static int
appendenv(const char *key, const char *value)
{
	char *curval, *newpair;
	int ret;

	curval = getenv(key);
	if (curval == NULL)
		ret = asprintf(&newpair, "%s=%s", key, value);
	else
		ret = asprintf(&newpair, "%s=%s:%s", key, curval, value);
	if (ret > 0)
		ret = putenv(newpair);
	if (ret < 0)
		warn("Failed to set environment variable: %s", key);
	return (ret);
}
#endif

static void
usage(void)
{

	fprintf(stderr,
	    "usage: stdbuf [-e 0|L|B|<sz>] [-i 0|L|B|<sz>] [-o 0|L|B|<sz>] "
	    "<cmd> [args ...]\n");
	exit(1);
}

int
main(int argc, char *argv[])
{
	char *ibuf, *obuf, *ebuf;
	int i;

	ibuf = obuf = ebuf = NULL;
	while ((i = getopt(argc, argv, "e:i:o:")) != -1) {
		switch (i) {
		case 'e':
			ebuf = optarg;
			break;
		case 'i':
			ibuf = optarg;
			break;
		case 'o':
			obuf = optarg;
			break;
		default:
			usage();
			break;
		}
	}
	argc -= optind;
	argv += optind;
	if (argc == 0)
		exit(0);

	if (ibuf != NULL && setenv("_STDBUF_I", ibuf, 1) == -1)
		warn("Failed to set environment variable: %s=%s",
		    "_STDBUF_I", ibuf);
	if (obuf != NULL && setenv("_STDBUF_O", obuf, 1) == -1)
		warn("Failed to set environment variable: %s=%s",
		    "_STDBUF_O", obuf);
	if (ebuf != NULL && setenv("_STDBUF_E", ebuf, 1) == -1)
		warn("Failed to set environment variable: %s=%s",
		    "_STDBUF_E", ebuf);

#ifndef __APPLE__
	appendenv("LD_PRELOAD", LIBSTDBUF);
	appendenv("LD_32_PRELOAD", LIBSTDBUF32);
#endif

	execvp(argv[0], argv);
	err(2, "%s", argv[0]);
}
