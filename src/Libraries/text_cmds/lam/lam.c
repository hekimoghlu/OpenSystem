/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
"@(#) Copyright (c) 1993\n\
	The Regents of the University of California.  All rights reserved.\n");
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)lam.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
__FBSDID("$FreeBSD$");

/*
 *	lam - laminate files
 *	Author:  John Kunze, UCB
 */

#ifndef __APPLE__
#include <sys/capsicum.h>

#include <capsicum_helpers.h>
#endif
#include <ctype.h>
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <unistd.h>

#define	MAXOFILES	20
#define	BIGBUFSIZ	5 * BUFSIZ

static struct openfile {	/* open file structure */
	FILE	*fp;		/* file pointer */
	short	eof;		/* eof flag */
	short	pad;		/* pad flag for missing columns */
	char	eol;		/* end of line character */
	const char *sepstring;	/* string to print before each line */
	const char *format;	/* printf(3) style string spec. */
}	input[MAXOFILES];

static int	morefiles;	/* set by getargs(), changed by gatherline() */
static int	nofinalnl;	/* normally append \n to each output line */
static char	line[BIGBUFSIZ];
static char	*linep;

static char    *gatherline(struct openfile *);
static void	getargs(char *[]);
static char    *pad(struct openfile *);
static void	usage(void);

int
main(int argc, char *argv[])
{
	struct	openfile *ip;

	if (argc == 1)
		usage();
#ifndef __APPLE__
	if (caph_limit_stdio() == -1)
		err(1, "unable to limit stdio");
#endif
	getargs(argv);
	if (!morefiles)
		usage();

#ifndef __APPLE__
	/*
	 * Cache NLS data, for strerror, for err(3), before entering capability
	 * mode.
	 */
	caph_cache_catpages();
	if (caph_enter() < 0)
		err(1, "unable to enter capability mode");
#endif

	for (;;) {
		linep = line;
		for (ip = input; ip->fp != NULL; ip++)
			linep = gatherline(ip);
		if (!morefiles)
			exit(0);
		fputs(line, stdout);
		fputs(ip->sepstring, stdout);
		if (!nofinalnl)
			putchar('\n');
	}
}

static void
getargs(char *av[])
{
	struct	openfile *ip = input;
	char *p, *c;
	static char fmtbuf[BUFSIZ];
	char *fmtp = fmtbuf;
	int P, S, F, T;
#ifndef __APPLE__
	cap_rights_t rights_ro;

	cap_rights_init(&rights_ro, CAP_READ, CAP_FSTAT);
#endif
	P = S = F = T = 0;		/* capitalized options */
	while ((p = *++av) != NULL) {
		if (*p != '-' || !p[1]) {
			if (++morefiles >= MAXOFILES)
				errx(1, "too many input files");
			if (*p == '-')
				ip->fp = stdin;
			else if ((ip->fp = fopen(p, "r")) == NULL) {
				err(1, "%s", p);
			}
#ifndef __APPLE__
			if (caph_rights_limit(fileno(ip->fp), &rights_ro) < 0)
				err(1, "unable to limit rights on: %s", p);
#endif
			ip->pad = P;
			if (!ip->sepstring)
				ip->sepstring = (S ? (ip-1)->sepstring : "");
			if (!ip->format)
				ip->format = ((P || F) ? (ip-1)->format : "%s");
			if (!ip->eol)
				ip->eol = (T ? (ip-1)->eol : '\n');
			ip++;
			continue;
		}
		c = ++p;
		switch (tolower((unsigned char)*c)) {
		case 's':
			if (*++p || (p = *++av))
				ip->sepstring = p;
			else
				usage();
			S = (*c == 'S' ? 1 : 0);
			break;
		case 't':
			if (*++p || (p = *++av))
				ip->eol = *p;
			else
				usage();
			T = (*c == 'T' ? 1 : 0);
			nofinalnl = 1;
			break;
		case 'p':
			ip->pad = 1;
			P = (*c == 'P' ? 1 : 0);
			/* FALLTHROUGH */
		case 'f':
			F = (*c == 'F' ? 1 : 0);
			if (*++p || (p = *++av)) {
				fmtp += strlen(fmtp) + 1;
				if (fmtp >= fmtbuf + sizeof(fmtbuf))
					errx(1, "no more format space");
				/* restrict format string to only valid width formatters */
				if (strspn(p, "-.0123456789") != strlen(p))
					errx(1, "invalid format string `%s'", p);
				if (snprintf(fmtp, fmtbuf + sizeof(fmtbuf) - fmtp, "%%%ss", p)
				    >= fmtbuf + sizeof(fmtbuf) - fmtp)
					errx(1, "no more format space");
				ip->format = fmtp;
			}
			else
				usage();
			break;
		default:
			usage();
		}
	}
	ip->fp = NULL;
	if (!ip->sepstring)
		ip->sepstring = "";
}

static char *
pad(struct openfile *ip)
{
	char *lp = linep;

	strlcpy(lp, ip->sepstring, line + sizeof(line) - lp);
	lp += strlen(lp);
	if (ip->pad) {
		snprintf(lp, line + sizeof(line) - lp, fmtcheck(ip->format, "%s"), "");
		lp += strlen(lp);
	}
	return (lp);
}

static char *
gatherline(struct openfile *ip)
{
	char s[BUFSIZ];
	int c;
	char *p;
	char *lp = linep;
	char *end = s + sizeof(s) - 1;

	if (ip->eof)
		return (pad(ip));
	for (p = s; (c = fgetc(ip->fp)) != EOF && p < end; p++)
		if ((*p = c) == ip->eol)
			break;
	*p = '\0';
	if (c == EOF) {
		ip->eof = 1;
		if (ferror(ip->fp)) {
			err(EX_IOERR, NULL);
		}
		if (ip->fp == stdin)
			fclose(stdin);
		morefiles--;
		return (pad(ip));
	}
	strlcpy(lp, ip->sepstring, line + sizeof(line) - lp);
	lp += strlen(lp);
	snprintf(lp, line + sizeof(line) - lp, fmtcheck(ip->format, "%s"), s);
	lp += strlen(lp);
	return (lp);
}

static void
usage(void)
{
	fprintf(stderr, "%s\n%s\n",
"usage: lam [ -f min.max ] [ -s sepstring ] [ -t c ] file ...",
"       lam [ -p min.max ] [ -s sepstring ] [ -t c ] file ...");
	exit(1);
}
