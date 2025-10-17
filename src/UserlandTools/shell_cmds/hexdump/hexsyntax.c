/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#if 0
static char sccsid[] = "@(#)hexsyntax.c	8.2 (Berkeley) 5/4/95";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hexdump.h"

off_t skip;				/* bytes to skip */

void
newsyntax(int argc, char ***argvp)
{
	int ch;
	char *p, **argv;

	argv = *argvp;
	if ((p = strrchr(argv[0], 'h')) != NULL &&
	    strcmp(p, "hd") == 0) {
		/* "Canonical" format, implies -C. */
		add("\"%08.8_Ax\n\"");
		add("\"%08.8_ax  \" 8/1 \"%02x \" \"  \" 8/1 \"%02x \" ");
		add("\"  |\" 16/1 \"%_p\" \"|\\n\"");
	}
	while ((ch = getopt(argc, argv, "bcCde:f:n:os:vx")) != -1)
		switch (ch) {
		case 'b':
			add("\"%07.7_Ax\n\"");
			add("\"%07.7_ax \" 16/1 \"%03o \" \"\\n\"");
			break;
		case 'c':
			add("\"%07.7_Ax\n\"");
			add("\"%07.7_ax \" 16/1 \"%3_c \" \"\\n\"");
			break;
		case 'C':
			add("\"%08.8_Ax\n\"");
			add("\"%08.8_ax  \" 8/1 \"%02x \" \"  \" 8/1 \"%02x \" ");
			add("\"  |\" 16/1 \"%_p\" \"|\\n\"");
			break;
		case 'd':
			add("\"%07.7_Ax\n\"");
			add("\"%07.7_ax \" 8/2 \"  %05u \" \"\\n\"");
			break;
		case 'e':
			add(optarg);
			break;
		case 'f':
			addfile(optarg);
			break;
		case 'n':
			if ((length = atoi(optarg)) < 0)
				errx(1, "%s: bad length value", optarg);
			break;
		case 'o':
			add("\"%07.7_Ax\n\"");
			add("\"%07.7_ax \" 8/2 \" %06o \" \"\\n\"");
			break;
		case 's':
			if ((skip = strtoll(optarg, &p, 0)) < 0)
				errx(1, "%s: bad skip value", optarg);
			switch(*p) {
			case 'b':
				skip *= 512;
				break;
			case 'k':
				skip *= 1024;
				break;
			case 'm':
				skip *= 1048576;
				break;
			case 'g':
				skip *= 1073741824;
				break;
			}
			break;
		case 'v':
			vflag = ALL;
			break;
		case 'x':
			add("\"%07.7_Ax\n\"");
			add("\"%07.7_ax \" 8/2 \"   %04x \" \"\\n\"");
			break;
		case '?':
			usage();
		}

	if (!fshead) {
		add("\"%07.7_Ax\n\"");
		add("\"%07.7_ax \" 8/2 \"%04x \" \"\\n\"");
	}

	*argvp += optind;
}

void
usage(void)
{
	(void)fprintf(stderr, "%s\n%s\n%s\n%s\n",
"usage: hexdump [-bcCdovx] [-e fmt] [-f fmt_file] [-n length]",
"               [-s skip] [file ...]",
"       hd      [-bcdovx]  [-e fmt] [-f fmt_file] [-n length]",
"               [-s skip] [file ...]");
	exit(1);
}
