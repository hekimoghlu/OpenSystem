/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef lint
__COPYRIGHT(
"@(#) Copyright (c) 1989, 1993\n\
	The Regents of the University of California.  All rights reserved.\n");
#endif /* not lint */

#if 0
#ifndef lint
static char sccsid[] = "@(#)paste.c	8.1 (Berkeley) 6/6/93";
#endif /* not lint */
#endif

__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <err.h>
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include <sysexits.h>

static wchar_t *delim;
static int delimcnt;

static int parallel(char **);
static int sequential(char **);
static int tr(wchar_t *);
static void usage(void);

static wchar_t tab[] = L"\t";

int
main(int argc, char *argv[])
{
	int ch, rval, seq;
	wchar_t *warg;
	const char *arg;
	size_t len;

	setlocale(LC_CTYPE, "");

	seq = 0;
	while ((ch = getopt(argc, argv, "d:s")) != -1)
		switch(ch) {
		case 'd':
			arg = optarg;
			len = mbsrtowcs(NULL, &arg, 0, NULL);
			if (len == (size_t)-1)
				err(1, "delimiters");
			warg = malloc((len + 1) * sizeof(*warg));
			if (warg == NULL)
				err(1, NULL);
			arg = optarg;
			len = mbsrtowcs(warg, &arg, len + 1, NULL);
			if (len == (size_t)-1)
				err(1, "delimiters");
			delimcnt = tr(delim = warg);
			break;
		case 's':
			seq = 1;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (*argv == NULL)
		usage();
	if (!delim) {
		delimcnt = 1;
		delim = tab;
	}

	if (seq)
		rval = sequential(argv);
	else
		rval = parallel(argv);
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(2, "stdout");
#endif
	exit(rval);
}

typedef struct _list {
	struct _list *next;
	FILE *fp;
	int cnt;
	char *name;
} LIST;

static int
parallel(char **argv)
{
	LIST *lp;
	int cnt;
	wint_t ich;
	wchar_t ch;
	char *p;
	LIST *head, *tmp;
	int opencnt, output;

	for (cnt = 0, head = tmp = NULL; (p = *argv); ++argv, ++cnt) {
		if ((lp = malloc(sizeof(LIST))) == NULL)
			err(1, NULL);
		if (p[0] == '-' && !p[1])
			lp->fp = stdin;
		else if (!(lp->fp = fopen(p, "r")))
			err(1, "%s", p);
		lp->next = NULL;
		lp->cnt = cnt;
		lp->name = p;
		if (!head)
			head = tmp = lp;
		else {
			tmp->next = lp;
			tmp = lp;
		}
	}

	for (opencnt = cnt; opencnt;) {
		for (output = 0, lp = head; lp; lp = lp->next) {
			if (!lp->fp) {
				if (output && lp->cnt &&
				    (ch = delim[(lp->cnt - 1) % delimcnt]))
					putwchar(ch);
				continue;
			}
			if ((ich = getwc(lp->fp)) == WEOF) {
				if (!--opencnt)
					break;
				lp->fp = NULL;
				if (output && lp->cnt &&
				    (ch = delim[(lp->cnt - 1) % delimcnt]))
					putwchar(ch);
				continue;
			}
			/*
			 * make sure that we don't print any delimiters
			 * unless there's a non-empty file.
			 */
			if (!output) {
				output = 1;
				for (cnt = 0; cnt < lp->cnt; ++cnt)
					if ((ch = delim[cnt % delimcnt]))
						putwchar(ch);
			} else if ((ch = delim[(lp->cnt - 1) % delimcnt]))
				putwchar(ch);
			if (ich == '\n')
				continue;
			do {
				putwchar(ich);
			} while ((ich = getwc(lp->fp)) != WEOF && ich != '\n');
			if (ferror(lp->fp)) {
				errx(EX_IOERR, "Error reading %s", lp->name);
			}
		}
		if (output)
			putwchar('\n');
	}

	return (0);
}

static int
sequential(char **argv)
{
	FILE *fp;
	int cnt, failed, needdelim;
	wint_t ch;
	char *p;

	failed = 0;
	for (; (p = *argv); ++argv) {
		if (p[0] == '-' && !p[1])
			fp = stdin;
		else if (!(fp = fopen(p, "r"))) {
			warn("%s", p);
			failed = 1;
			continue;
		}
		cnt = needdelim = 0;
		while ((ch = getwc(fp)) != WEOF) {
			if (needdelim) {
				needdelim = 0;
				if (delim[cnt] != '\0')
					putwchar(delim[cnt]);
				if (++cnt == delimcnt)
					cnt = 0;
			}
			if (ch != '\n')
				putwchar(ch);
			else
				needdelim = 1;
		}
		if (needdelim)
			putwchar('\n');
		if (fp != stdin)
			(void)fclose(fp);
	}

	return (failed != 0);
}

static int
tr(wchar_t *arg)
{
	int cnt;
	wchar_t ch, *p;

	for (p = arg, cnt = 0; (ch = *p++); ++arg, ++cnt)
		if (ch == '\\')
			switch(ch = *p++) {
			case 'n':
				*arg = '\n';
				break;
			case 't':
				*arg = '\t';
				break;
			case '0':
				*arg = '\0';
				break;
			default:
				*arg = ch;
				break;
		} else
			*arg = ch;

	if (!cnt)
		errx(1, "no delimiters specified");
	return(cnt);
}

static void
usage(void)
{
	(void)fprintf(stderr, "usage: paste [-s] [-d delimiters] file ...\n");
	exit(1);
}
