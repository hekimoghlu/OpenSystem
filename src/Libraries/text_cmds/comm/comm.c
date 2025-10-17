/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#ifndef lint
static const char copyright[] =
"@(#) Copyright (c) 1989, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#if 0
#ifndef lint
static char sccsid[] = "From: @(#)comm.c	8.4 (Berkeley) 5/4/95";
#endif
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <err.h>
#include <limits.h>
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>

static int iflag;
static const char *tabs[] = { "", "\t", "\t\t" };

static FILE	*file(const char *);
static wchar_t	*convert(const char *);
static void	show(FILE *, const char *, const char *, char **, size_t *);
static void	usage(void);

int
main(int argc, char *argv[])
{
	int comp, read1, read2;
	int ch, flag1, flag2, flag3;
	FILE *fp1, *fp2;
	const char *col1, *col2, *col3;
	size_t line1len, line2len;
	char *line1, *line2;
	ssize_t n1, n2;
	wchar_t *tline1, *tline2;
	const char **p;

	(void) setlocale(LC_ALL, "");

	flag1 = flag2 = flag3 = 1;

	while ((ch = getopt(argc, argv, "123i")) != -1)
		switch(ch) {
		case '1':
			flag1 = 0;
			break;
		case '2':
			flag2 = 0;
			break;
		case '3':
			flag3 = 0;
			break;
		case 'i':
			iflag = 1;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc != 2 || !argv[0] || !argv[1])
		usage();

	fp1 = file(argv[0]);
	fp2 = file(argv[1]);

	/* for each column printed, add another tab offset */
	p = tabs;
	col1 = col2 = col3 = NULL;
	if (flag1)
		col1 = *p++;
	if (flag2)
		col2 = *p++;
	if (flag3)
		col3 = *p;

	line1len = line2len = 0;
	line1 = line2 = NULL;
	n1 = n2 = -1;

	for (read1 = read2 = 1;;) {
		/* read next line, check for EOF */
		if (read1) {
			n1 = getline(&line1, &line1len, fp1);
			if (n1 < 0 && ferror(fp1))
				err(1, "%s", argv[0]);
			if (n1 > 0 && line1[n1 - 1] == '\n')
				line1[n1 - 1] = '\0';

		}
		if (read2) {
			n2 = getline(&line2, &line2len, fp2);
			if (n2 < 0 && ferror(fp2))
				err(1, "%s", argv[1]);
			if (n2 > 0 && line2[n2 - 1] == '\n')
				line2[n2 - 1] = '\0';
		}

		/* if one file done, display the rest of the other file */
		if (n1 < 0) {
			if (n2 >= 0)
				show(fp2, argv[1], col2, &line2, &line2len);
			break;
		}
		if (n2 < 0) {
			if (n1 >= 0)
				show(fp1, argv[0], col1, &line1, &line1len);
			break;
		}

		tline2 = NULL;
		if ((tline1 = convert(line1)) != NULL)
			tline2 = convert(line2);
		if (tline1 == NULL || tline2 == NULL)
			comp = strcmp(line1, line2);
		else
			comp = wcscoll(tline1, tline2);
		if (tline1 != NULL)
			free(tline1);
		if (tline2 != NULL)
			free(tline2);

		/* lines are the same */
		if (!comp) {
			read1 = read2 = 1;
			if (col3 != NULL)
				(void)printf("%s%s\n", col3, line1);
			continue;
		}

		/* lines are different */
		if (comp < 0) {
			read1 = 1;
			read2 = 0;
			if (col1 != NULL)
				(void)printf("%s%s\n", col1, line1);
		} else {
			read1 = 0;
			read2 = 1;
			if (col2 != NULL)
				(void)printf("%s%s\n", col2, line2);
		}
	}
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(0);
}

static wchar_t *
convert(const char *str)
{
	size_t n;
	wchar_t *buf, *p;

	if ((n = mbstowcs(NULL, str, 0)) == (size_t)-1)
		return (NULL);
	if (SIZE_MAX / sizeof(*buf) < n + 1)
		errx(1, "conversion buffer length overflow");
	if ((buf = malloc((n + 1) * sizeof(*buf))) == NULL)
		err(1, "malloc");
	if (mbstowcs(buf, str, n + 1) != n)
		errx(1, "internal mbstowcs() error");

	if (iflag) {
		for (p = buf; *p != L'\0'; p++)
			*p = towlower(*p);
	}

	return (buf);
}

static void
show(FILE *fp, const char *fn, const char *offset, char **bufp, size_t *buflenp)
{
	ssize_t n;

	do {
		/* offset is NULL when draining fp, not printing (rdar://89062040) */
		if (offset != NULL)
			(void)printf("%s%s\n", offset, *bufp);
		if ((n = getline(bufp, buflenp, fp)) < 0)
			break;
		if (n > 0 && offset != NULL && (*bufp)[n - 1] == '\n')
			(*bufp)[n - 1] = '\0';
	} while (1);
	if (ferror(fp))
		err(1, "%s", fn);
}

static FILE *
file(const char *name)
{
	FILE *fp;

	if (!strcmp(name, "-"))
		return (stdin);
	if ((fp = fopen(name, "r")) == NULL) {
		err(1, "%s", name);
	}
	return (fp);
}

static void
usage(void)
{
	(void)fprintf(stderr, "usage: comm [-123i] file1 file2\n");
	exit(1);
}
