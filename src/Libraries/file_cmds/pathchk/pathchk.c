/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
 * pathchk -- check pathnames
 *
 * Check whether files could be created with the names specified on the
 * command line. If -p is specified, check whether the pathname is portable
 * to all POSIX systems.
 */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/stat.h>

#include <err.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int	 check(const char *);
static int	 portable(const char *);
static void	 usage(void);

static int	 pflag;			/* Perform portability checks */
static int	 Pflag;			/* Check for empty paths, leading '-' */

int
main(int argc, char *argv[])
{
	int ch, rval;
	const char *arg;

	while ((ch = getopt(argc, argv, "pP")) > 0) {
		switch (ch) {
		case 'p':
			pflag = 1;
			break;
		case 'P':
			Pflag = 1;
			break;
		default:
			usage();
			/*NOTREACHED*/
		}
	}
	argc -= optind;
	argv += optind;

	if (argc == 0)
		usage();

	rval = 0;
	while ((arg = *argv++) != NULL)
		rval |= check(arg);

	exit(rval);
}

static void
usage(void)
{

	fprintf(stderr, "usage: pathchk [-Pp] pathname ...\n");
	exit(1);
}

static int
check(const char *path)
{
	struct stat sb;
	long complen, namemax, pathmax, svnamemax;
	int last;
	char *end, *p, *pathd;

	if ((pathd = strdup(path)) == NULL)
		err(1, "strdup");

	p = pathd;

	if (Pflag && *p == '\0') {
		warnx("%s: empty pathname", path);
		goto bad;
	}
	if ((Pflag || pflag) && (*p == '-' || strstr(p, "/-") != NULL)) {
		warnx("%s: contains a component starting with '-'", path);
		goto bad;
	}

	if (!pflag) {
		errno = 0;
		namemax = pathconf(*p == '/' ? "/" : ".", _PC_NAME_MAX);
		if (namemax == -1 && errno != 0)
			namemax = NAME_MAX;
	} else
		namemax = _POSIX_NAME_MAX;

	for (;;) {
		p += strspn(p, "/");
		complen = (long)strcspn(p, "/");
		end = p + complen;
		last = *end == '\0';
		*end = '\0';

		if (namemax != -1 && complen > namemax) {
			warnx("%s: %s: component too long (limit %ld)", path,
			    p, namemax);
			goto bad;
		}

		if (!pflag && stat(pathd, &sb) == -1 && errno != ENOENT) {
			warn("%s: %.*s", path, (int)(strlen(pathd) -
			    complen - 1), pathd);
			goto bad;
		}

		if (pflag && !portable(p)) {
			warnx("%s: %s: component contains non-portable "
			    "character", path, p);
			goto bad;
		}

		if (last)
			break;

		if (!pflag) {
			errno = 0;
			svnamemax = namemax;
			namemax = pathconf(pathd, _PC_NAME_MAX);
			if (namemax == -1 && errno != 0)
				namemax = svnamemax;
		}

		*end = '/';
		p = end + 1;
	}

	if (!pflag) {
		errno = 0;
		pathmax = pathconf(path, _PC_PATH_MAX);
		if (pathmax == -1 && errno != 0)
			pathmax = PATH_MAX;
	} else
		pathmax = _POSIX_PATH_MAX;
	if (pathmax != -1 && strlen(path) >= (size_t)pathmax) {
		warnx("%s: path too long (limit %ld)", path, pathmax - 1);
		goto bad;
	}

	free(pathd);
	return (0);

bad:	free(pathd);
	return (1);
}

/*
 * Check whether a path component contains only portable characters.
 */
static int
portable(const char *path)
{
	static const char charset[] =
	    "abcdefghijklmnopqrstuvwxyz"
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	    "0123456789._-";
	long s;

	s = strspn(path, charset);
	if (path[s] != '\0')
		return (0);

	return (1);
}
