/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
/*-
 * Plug-compatible replacement for getopt() for parsing tar-like
 * arguments.  If the first argument begins with "-", it uses getopt;
 * otherwise, it uses the old rules used by tar, dump, and ps.
 *
 * Written 25 August 1985 by John Gilmore (ihnp4!hoptoad!gnu) and placed
 * in the Public Domain for your edification and enjoyment.
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int getoldopt(int, char **, const char *);

int
getoldopt(int argc, char **argv, const char *optstring)
{
	static char	*key;		/* Points to next keyletter */
	static char	use_getopt;	/* !=0 if argv[1][0] was '-' */
	char		c;
	char		*place;

	optarg = NULL;

	if (key == NULL) {		/* First time */
		if (argc < 2)
			return (-1);
		key = argv[1];
		if (*key == '-')
			use_getopt++;
		else
			optind = 2;
	}

	if (use_getopt)
		return (getopt(argc, argv, optstring));

	c = *key++;
	if (c == '\0') {
		key--;
		return (-1);
	}
	place = strchr(optstring, c);

	if (place == NULL || c == ':') {
		fprintf(stderr, "%s: unknown option %c\n", argv[0], c);
		return ('?');
	}

	place++;
	if (*place == ':') {
		if (optind < argc) {
			optarg = argv[optind];
			optind++;
		} else {
			fprintf(stderr, "%s: %c argument missing\n",
				argv[0], c);
			return ('?');
		}
	}

	return (c);
}

