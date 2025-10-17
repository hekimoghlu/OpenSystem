/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#if 0
#ifndef lint
static char copyright[] =
"@(#) Copyright (c) 1989, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)locate.code.c	8.1 (Berkeley) 6/6/93";
#endif /* not lint */
#endif

/*
 * PURPOSE:	sorted list compressor (works with a modified 'find'
 *		to encode/decode a filename database)
 *
 * USAGE:	bigram < list > bigrams
 *		process bigrams (see updatedb) > common_bigrams
 *		code common_bigrams < list > squozen_list
 *
 * METHOD:	Uses 'front compression' (see ";login:", Volume 8, Number 1
 *		February/March 1983, p. 8).  Output format is, per line, an
 *		offset differential count byte followed by a partially bigram-
 *		encoded ascii residue.  A bigram is a two-character sequence,
 *		the first 128 most common of which are encoded in one byte.
 *
 * EXAMPLE:	For simple front compression with no bigram encoding,
 *		if the input is...		then the output is...
 *
 *		/usr/src			 0 /usr/src
 *		/usr/src/cmd/aardvark.c		 8 /cmd/aardvark.c
 *		/usr/src/cmd/armadillo.c	14 armadillo.c
 *		/usr/tmp/zoo			 5 tmp/zoo
 *
 *	The codes are:
 *
 *	0-28	likeliest differential counts + offset to make nonnegative
 *	30	switch code for out-of-range count to follow in next word
 *      31      an 8 bit char followed
 *	128-255 bigram codes (128 most common, as determined by 'updatedb')
 *	32-127  single character (printable) ascii residue (ie, literal)
 *
 * The locate database store any character except newline ('\n') 
 * and NUL ('\0'). The 8-bit character support don't wast extra
 * space until you have characters in file names less than 32
 * or greather than 127.
 * 
 *
 * SEE ALSO:	updatedb.sh, ../bigram/locate.bigram.c
 *
 * AUTHOR:	James A. Woods, Informatics General Corp.,
 *		NASA Ames Research Center, 10/82
 *              8-bit file names characters: 
 *              	Wolfram Schneider, Berlin September 1996
 */

#include <sys/param.h>
#include <err.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "locate.h"

#define	BGBUFSIZE	(NBG * 2)	/* size of bigram buffer */

u_char buf1[MAXPATHLEN] = " ";	
u_char buf2[MAXPATHLEN];
u_char bigrams[BGBUFSIZE + 1] = { 0 };

#define LOOKUP 1 /* use a lookup array instead a function, 3x faster */

#ifdef LOOKUP
#define BGINDEX(x) (big[(u_char)*x][(u_char)*(x + 1)])
typedef short bg_t;
bg_t big[UCHAR_MAX + 1][UCHAR_MAX + 1];
#else
#define BGINDEX(x) bgindex(x)
typedef int bg_t;
int	bgindex(char *);
#endif /* LOOKUP */


void	usage(void);

int
main(int argc, char *argv[])
{
	u_char *cp, *oldpath, *path;
	int ch, code, count, diffcount, oldcount;
	u_int i, j;
	FILE *fp;

	while ((ch = getopt(argc, argv, "")) != -1)
		switch(ch) {
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc != 1)
		usage();

	if ((fp = fopen(argv[0], "r")) == NULL)
		err(1, "%s", argv[0]);

	/* First copy bigram array to stdout. */
	(void)fgets(bigrams, BGBUFSIZE + 1, fp);

	if (fwrite(bigrams, 1, BGBUFSIZE, stdout) != BGBUFSIZE)
		err(1, "stdout");
	(void)fclose(fp);

#ifdef LOOKUP
	/* init lookup table */
	for (i = 0; i < UCHAR_MAX + 1; i++)
	    	for (j = 0; j < UCHAR_MAX + 1; j++) 
			big[i][j] = (bg_t)-1;

	for (cp = bigrams, i = 0; *cp != '\0'; i += 2, cp += 2)
	        big[(u_char)*cp][(u_char)*(cp + 1)] = (bg_t)i;

#endif /* LOOKUP */

	oldpath = buf1;
	path = buf2;
	oldcount = 0;

	while (fgets(path, sizeof(buf2), stdin) != NULL) {

		/* skip empty lines */
		if (*path == '\n')
			continue;

		/* remove newline */
		for (cp = path; *cp != '\0'; cp++) {
#ifndef LOCATE_CHAR30
			/* old locate implementations core'd for char 30 */
			if (*cp == SWITCH)
				*cp = '?';
			else
#endif /* !LOCATE_CHAR30 */

			/* chop newline */
			if (*cp == '\n')
				*cp = '\0';
		}

		/* Skip longest common prefix. */
		for (cp = path; *cp == *oldpath; cp++, oldpath++)
			if (*cp == '\0')
				break;

		count = cp - path;
		diffcount = count - oldcount + OFFSET;
		oldcount = count;
		if (diffcount < 0 || diffcount > 2 * OFFSET) {
			if (putchar(SWITCH) == EOF ||
			    putw(diffcount, stdout) == EOF)
				err(1, "stdout");
		} else
			if (putchar(diffcount) == EOF)
				err(1, "stdout");

		while (*cp != '\0') {
			/* print *two* characters */

			if ((code = BGINDEX(cp)) != (bg_t)-1) {
				/*
				 * print *one* as bigram
				 * Found, so mark byte with 
				 *  parity bit. 
				 */
				if (putchar((code / 2) | PARITY) == EOF)
					err(1, "stdout");
				cp += 2;
			}

			else {
				for (i = 0; i < 2; i++) {
					if (*cp == '\0')
						break;

					/* print umlauts in file names */
					if (*cp < ASCII_MIN || 
					    *cp > ASCII_MAX) {
						if (putchar(UMLAUT) == EOF ||
						    putchar(*cp++) == EOF)
							err(1, "stdout");
					} 

					else {
						/* normal character */
						if(putchar(*cp++) == EOF)
							err(1, "stdout");
					}
				}

			}
		}

		if (path == buf1) {		/* swap pointers */
			path = buf2;
			oldpath = buf1;
		} else {
			path = buf1;
			oldpath = buf2;
		}
	}
	/* Non-zero status if there were errors */
	if (fflush(stdout) != 0 || ferror(stdout))
		exit(1);
	exit(0);
}

#ifndef LOOKUP
int
bgindex(char *bg)		/* Return location of bg in bigrams or -1. */
{
	char bg0, bg1, *p;

	bg0 = bg[0];
	bg1 = bg[1];
	for (p = bigrams; *p != NULL; p++)
		if (*p++ == bg0 && *p == bg1)
			break;
	return (*p == NULL ? -1 : (--p - bigrams));
}
#endif /* !LOOKUP */

void
usage(void)
{
	(void)fprintf(stderr,
	    "usage: locate.code common_bigrams < list > squozen_list\n");
	exit(1);
}
