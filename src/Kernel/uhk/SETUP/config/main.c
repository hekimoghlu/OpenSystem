/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
 * Mach Operating System
 * Copyright (c) 1990 Carnegie-Mellon University
 * Copyright (c) 1989 Carnegie-Mellon University
 * Copyright (c) 1988 Carnegie-Mellon University
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */

/*
 * Copyright (c) 1980 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, Berkeley.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef lint
char copyright[] =
    "@(#) Copyright (c) 1980 Regents of the University of California.\n\
 All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] __attribute__((used)) = "@(#)main.c	5.9 (Berkeley) 6/18/88";
#endif /* not lint */

#include <stdio.h>
#include <ctype.h>
#include "parser.h"
#include "config.h"

/*
 * Config builds a set of files for building a UNIX
 * system given a description of the desired system.
 */
int
main(int argc, char *argv[])
{
	source_directory = "..";        /* default */
	object_directory = "..";
	config_directory = (char *) 0;
	while ((argc > 1) && (argv[1][0] == '-')) {
		char            *c;

		argv++; argc--;
		for (c = &argv[0][1]; *c; c++) {
			switch (*c) {
			case 'b':
				build_directory = argv[1];
				goto check_arg;

			case 'd':
				source_directory = argv[1];
				goto check_arg;

			case 'o':
				object_directory = argv[1];
				goto check_arg;

			case 'c':
				config_directory = argv[1];

check_arg:
				if (argv[1] == (char *) 0) {
					goto usage_error;
				}
				argv++; argc--;
				break;

			case 'p':
				profiling++;
				break;
			default:
				goto usage_error;
			}
		}
	}
	if (config_directory == (char *) 0) {
		config_directory =
		    malloc((unsigned) strlen(source_directory) + 6);
		(void) sprintf(config_directory, "%s/conf", source_directory);
	}
	if (argc != 2) {
usage_error:    ;
		fprintf(stderr, "usage: config [ -bcdo dir ] [ -p ] sysname\n");
		exit(1);
	}
	if (!build_directory) {
		build_directory = argv[1];
	}
	if (freopen(argv[1], "r", stdin) == NULL) {
		perror(argv[1]);
		exit(2);
	}
	dtab = NULL;
	confp = &conf_list;
	opt = 0;
	if (yyparse()) {
		exit(3);
	}

	mkioconf();                     /* ioconf.c */
	makefile();                     /* build Makefile */
	headers();                      /* make a lot of .h files */

	return 0;
}

/*
 * get_word
 *	returns EOF on end of file
 *	NULL on end of line
 *	pointer to the word otherwise
 */
const char *
get_word(FILE *fp)
{
	static char line[80];
	int ch;
	char *cp;

	while ((ch = getc(fp)) != EOF) {
		if (ch != ' ' && ch != '\t') {
			break;
		}
	}
	if (ch == EOF) {
		return (char *)EOF;
	}
	if (ch == '\n') {
		return NULL;
	}
	if (ch == '|') {
		return "|";
	}
	cp = line;
	*cp++ = ch;
	while ((ch = getc(fp)) != EOF) {
		if (isspace(ch)) {
			break;
		}
		*cp++ = ch;
	}
	*cp = 0;
	if (ch == EOF) {
		return (char *)EOF;
	}
	(void) ungetc(ch, fp);
	return line;
}

/*
 * get_rest
 *	returns EOF on end of file
 *	NULL on end of line
 *	pointer to the word otherwise
 */
char *
get_rest(FILE *fp)
{
	static char line[80];
	int ch;
	char *cp;

	cp = line;
	while ((ch = getc(fp)) != EOF) {
		if (ch == '\n') {
			break;
		}
		*cp++ = ch;
	}
	*cp = 0;
	if (ch == EOF) {
		return (char *)EOF;
	}
	return line;
}

/*
 * prepend the path to a filename
 */
char *
path(const char *file)
{
	char *cp;

	cp = malloc((unsigned)(strlen(build_directory) +
	    strlen(file) +
	    strlen(object_directory) +
	    3));
	(void) sprintf(cp, "%s/%s/%s", object_directory, build_directory, file);
	return cp;
}
