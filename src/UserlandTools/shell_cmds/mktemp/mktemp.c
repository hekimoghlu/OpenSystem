/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
/*
 * This program was originally written long ago, originally for a non
 * BSD-like OS without mkstemp().  It's been modified over the years
 * to use mkstemp() rather than the original O_CREAT|O_EXCL/fstat/lstat
 * etc style hacks.
 * A cleanup, misc options and mkdtemp() calls were added to try and work
 * more like the OpenBSD version - which was first to publish the interface.
 */

#include <err.h>
#include <getopt.h>
#ifdef __APPLE__
#include <limits.h>
#endif
#include <paths.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef lint
static const char rcsid[] =
	"$FreeBSD$";
#endif /* not lint */

static void usage(void);

static const struct option long_opts[] = {
	{"directory",	no_argument,	NULL,	'd'},
	{"tmpdir",	optional_argument,	NULL,	'p'},
	{"quiet",	no_argument,	NULL,	'q'},
	{"dry-run",	no_argument,	NULL,	'u'},
	{NULL,		no_argument,	NULL,	0},
};

int
main(int argc, char **argv)
{
	int c, fd, ret;
	const char *prefix, *tmpdir;
	char *name;
	int dflag, qflag, tflag, uflag;
#ifdef __APPLE__
	char tmpbuf[PATH_MAX];
#endif
	bool prefer_tmpdir;

	ret = dflag = qflag = tflag = uflag = 0;
	prefer_tmpdir = true;
	prefix = "mktemp";
	name = NULL;
	tmpdir = NULL;

	while ((c = getopt_long(argc, argv, "dp:qt:u", long_opts, NULL)) != -1)
		switch (c) {
		case 'd':
			dflag++;
			break;

		case 'p':
			tmpdir = optarg;
			if (tmpdir == NULL || *tmpdir == '\0')
				tmpdir = getenv("TMPDIR");
#ifdef __APPLE__
			if (tmpdir == NULL && confstr(_CS_DARWIN_USER_TEMP_DIR,
			    tmpbuf, sizeof(tmpbuf)) > 0)
				tmpdir = tmpbuf;
#endif

			/*
			 * We've already done the necessary environment
			 * fallback, skip the later one.
			 */
			prefer_tmpdir = false;
			break;

		case 'q':
			qflag++;
			break;

		case 't':
			prefix = optarg;
			tflag++;
			break;

		case 'u':
			uflag++;
			break;

		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	if (!tflag && argc < 1) {
		tflag = 1;
		prefix = "tmp";

		/*
		 * For this implied -t mode, we actually want to swap the usual
		 * order of precedence: -p, then TMPDIR, then /tmp.
		 */
		prefer_tmpdir = false;
	}

	if (tflag) {
		const char *envtmp;
		size_t len;

		envtmp = NULL;

		/*
		 * $TMPDIR preferred over `-p` if specified, for compatibility.
		 */
#ifdef __APPLE__
		if (prefer_tmpdir || tmpdir == NULL) {
			if (confstr(_CS_DARWIN_USER_TEMP_DIR, tmpbuf,
			    sizeof(tmpbuf)) > 0) {
				envtmp = tmpbuf;
			} else {
				envtmp = getenv("TMPDIR");
			}
		}
#else
		if (prefer_tmpdir || tmpdir == NULL)
			envtmp = getenv("TMPDIR");
#endif
		if (envtmp != NULL)
			tmpdir = envtmp;
		if (tmpdir == NULL)
			tmpdir = _PATH_TMP;
		len = strlen(tmpdir);
		if (len > 0 && tmpdir[len - 1] == '/')
			asprintf(&name, "%s%s.XXXXXXXXXX", tmpdir, prefix);
		else
			asprintf(&name, "%s/%s.XXXXXXXXXX", tmpdir, prefix);
		/* if this fails, the program is in big trouble already */
		if (name == NULL) {
			if (qflag)
				return (1);
			else
				errx(1, "cannot generate template");
		}
	}

	/* generate all requested files */
	while (name != NULL || argc > 0) {
		if (name == NULL) {
			if (!tflag && tmpdir != NULL)
				asprintf(&name, "%s/%s", tmpdir, argv[0]);
			else
				name = strdup(argv[0]);
			if (name == NULL)
				err(1, "%s", argv[0]);
			argv++;
			argc--;
		}

		if (dflag) {
			if (mkdtemp(name) == NULL) {
				ret = 1;
				if (!qflag)
					warn("mkdtemp failed on %s", name);
			} else {
				printf("%s\n", name);
				if (uflag)
					rmdir(name);
			}
		} else {
			fd = mkstemp(name);
			if (fd < 0) {
				ret = 1;
				if (!qflag)
					warn("mkstemp failed on %s", name);
			} else {
				close(fd);
				if (uflag)
					unlink(name);
				printf("%s\n", name);
			}
		}
		if (name)
			free(name);
		name = NULL;
	}
	return (ret);
}

static void
usage(void)
{
	fprintf(stderr,
		"usage: mktemp [-d] [-p tmpdir] [-q] [-t prefix] [-u] template "
		"...\n");
	fprintf(stderr,
		"       mktemp [-d] [-p tmpdir] [-q] [-u] -t prefix \n");
	exit (1);
}
