/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

#include <sys/types.h>

#include <err.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sysexits.h>
#include <unistd.h>

#include "getconf.h"

static void	do_allsys(void);
static void	do_allpath(const char *path);
static void	do_confstr(const char *name, int key);
static void	do_sysconf(const char *name, int key);
static void	do_pathconf(const char *name, int key, const char *path);

static void
usage(void)
{
	fprintf(stderr,
"usage: getconf -a [pathname]\n"
"       getconf [-v prog_env] system_var\n"
"       getconf [-v prog_env] path_var pathname\n");
	exit(EX_USAGE);
}

int
main(int argc, char **argv)
{
	bool aflag;
	int c, key, valid;
	const char *name, *vflag, *alt_path;
	intmax_t limitval;
	uintmax_t ulimitval;

	aflag = false;
	vflag = NULL;
	while ((c = getopt(argc, argv, "av:")) != -1) {
		switch (c) {
		case 'a':
			aflag = true;
			break;
		case 'v':
			vflag = optarg;
			break;

		default:
			usage();
		}
	}

	if (aflag) {
		if (vflag != NULL)
			usage();
		if (argv[optind] == NULL)
			do_allsys();
		else
			do_allpath(argv[optind]);
		return (0);
	}

	if ((name = argv[optind]) == NULL)
		usage();

	if (vflag != NULL) {
		if ((valid = find_progenv(vflag, &alt_path)) == 0)
			errx(EX_USAGE, "invalid programming environment %s",
			     vflag);
		if (valid > 0 && alt_path != NULL) {
			if (argv[optind + 1] == NULL)
				execl(alt_path, "getconf", argv[optind],
				      (char *)NULL);
			else
				execl(alt_path, "getconf", argv[optind],
				      argv[optind + 1], (char *)NULL);

			err(EX_OSERR, "execl: %s", alt_path);
		}
		if (valid < 0)
			errx(EX_UNAVAILABLE, "environment %s is not available",
			     vflag);
	}

	if (argv[optind + 1] == NULL) { /* confstr or sysconf */
		if ((valid = find_unsigned_limit(name, &ulimitval)) != 0) {
			if (valid > 0)
				printf("%" PRIuMAX "\n", ulimitval);
			else
				printf("undefined\n");
#ifdef __APPLE__
			if (ferror(stdout) != 0 || fflush(stdout) != 0)
				err(1, "stdout");
#endif
			return 0;
		}
		if ((valid = find_limit(name, &limitval)) != 0) {
			if (valid > 0)
				printf("%" PRIdMAX "\n", limitval);
			else
				printf("undefined\n");

#ifdef __APPLE__
			if (ferror(stdout) != 0 || fflush(stdout) != 0)
				err(1, "stdout");
#endif
			return 0;
		}
		if ((valid = find_confstr(name, &key)) != 0) {
			if (valid > 0)
				do_confstr(name, key);
			else
				printf("undefined\n");
		} else {
			valid = find_sysconf(name, &key);
			if (valid > 0) {
				do_sysconf(name, key);
			} else if (valid < 0) {
				printf("undefined\n");
			} else
				errx(EX_USAGE,
				     "no such configuration parameter `%s'",
				     name);
		}
	} else {
		valid = find_pathconf(name, &key);
		if (valid != 0) {
			if (valid > 0)
				do_pathconf(name, key, argv[optind + 1]);
			else
				printf("undefined\n");
		} else
			errx(EX_USAGE,
			     "no such path configuration parameter `%s'",
			     name);
	}
#ifdef __APPLE__
			if (ferror(stdout) != 0 || fflush(stdout) != 0)
				err(1, "stdout");
#endif
	return 0;
}

static void
do_onestr(const char *name, int key)
{
	size_t len;

	errno = 0;
	len = confstr(key, 0, 0);
	if (len == 0 && errno != 0) {
		warn("confstr: %s", name);
		return;
	}
	printf("%s: ", name);
	if (len == 0)
		printf("undefined\n");
	else {
		char buf[len + 1];

		confstr(key, buf, len);
		printf("%s\n", buf);
	}
}

static void
do_onesys(const char *name, int key)
{
	long value;

	errno = 0;
	value = sysconf(key);
	if (value == -1 && errno != 0) {
		warn("sysconf: %s", name);
		return;
	}
	printf("%s: ", name);
	if (value == -1)
		printf("undefined\n");
	else
		printf("%ld\n", value);
}

static void
do_allsys(void)
{

	foreach_confstr(do_onestr);
	foreach_sysconf(do_onesys);
}

static void
do_onepath(const char *name, int key, const char *path)
{
	long value;

	errno = 0;
	value = pathconf(path, key);
	if (value == -1 && errno != EINVAL && errno != 0)
		warn("pathconf: %s", name);
	printf("%s: ", name);
	if (value == -1)
		printf("undefined\n");
	else
		printf("%ld\n", value);
}

static void
do_allpath(const char *path)
{

	foreach_pathconf(do_onepath, path);
}

static void
do_confstr(const char *name, int key)
{
	size_t len;
	int savederr;

	savederr = errno;
	errno = 0;
	len = confstr(key, 0, 0);
	if (len == 0) {
		if (errno)
			err(EX_OSERR, "confstr: %s", name);
		else
			printf("undefined\n");
	} else {
		char buf[len + 1];

		confstr(key, buf, len);
		printf("%s\n", buf);
	}
	errno = savederr;
}

static void
do_sysconf(const char *name, int key)
{
	long value;

	errno = 0;
	value = sysconf(key);
	if (value == -1 && errno != 0)
		err(EX_OSERR, "sysconf: %s", name);
	else if (value == -1)
		printf("undefined\n");
	else
		printf("%ld\n", value);
}

static void
do_pathconf(const char *name, int key, const char *path)
{
	long value;

	errno = 0;
	value = pathconf(path, key);
	if (value == -1 && errno != 0)
		err(EX_OSERR, "pathconf: %s", name);
	else if (value == -1)
		printf("undefined\n");
	else
		printf("%ld\n", value);
}

