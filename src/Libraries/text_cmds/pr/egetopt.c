/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
static char sccsid[] = "@(#)egetopt.c	8.1 (Berkeley) 6/6/93";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "extern.h"

/*
 * egetopt:	get option letter from argument vector (an extended
 *		version of getopt).
 *
 * Non standard additions to the ostr specs are:
 * 1) '?': immediate value following arg is optional (no white space
 *    between the arg and the value)
 * 2) '#': +/- followed by a number (with an optional sign but
 *    no white space between the arg and the number). The - may be
 *    combined with other options, but the + cannot.
 */

int	eopterr = 1;		/* if error message should be printed */
int	eoptind = 1;		/* index into parent argv vector */
int	eoptopt;		/* character checked for validity */
char	*eoptarg;		/* argument associated with option */

#define	BADCH	(int)'?'

static char	emsg[] = "";

int
egetopt(int nargc, char * const *nargv, const char *ostr)
{
	static char *place = emsg;	/* option letter processing */
	char *oli;			/* option letter list index */
	static int delim;		/* which option delimiter */
	char *p;
	static char savec = '\0';

	if (savec != '\0') {
		*place = savec;
		savec = '\0';
	}

	if (!*place) {
		/*
		 * update scanning pointer
		 */
		if ((eoptind >= nargc) ||
		    ((*(place = nargv[eoptind]) != '-') && (*place != '+'))) {
			place = emsg;
			return (-1);
		}

		delim = (int)*place;
		if (place[1] && *++place == '-' && !place[1]) {
			/*
			 * found "--"
			 */
			++eoptind;
			place = emsg;
			return (-1);
		}
	}

	/*
	 * check option letter
	 */
	if ((eoptopt = (int)*place++) == (int)':' || (eoptopt == (int)'?') ||
	    !(oli = strchr(ostr, eoptopt))) {
		/*
		 * if the user didn't specify '-' as an option,
		 * assume it means -1 when by itself.
		 */
		if ((eoptopt == (int)'-') && !*place)
			return (-1);
		if (strchr(ostr, '#') && (isdigit(eoptopt) ||
		    (((eoptopt == (int)'-') || (eoptopt == (int)'+')) &&
		      isdigit(*place)))) {
			/*
			 * # option: +/- with a number is ok
			 */
			for (p = place; *p != '\0'; ++p) {
				if (!isdigit(*p))
					break;
			}
			eoptarg = place-1;

			if (*p == '\0') {
				place = emsg;
				++eoptind;
			} else {
				place = p;
				savec = *p;
				*place = '\0';
			}
			return (delim);
		}

		if (!*place)
			++eoptind;
		if (eopterr) {
			if (!(p = strrchr(*nargv, '/')))
				p = *nargv;
			else
				++p;
			(void)fprintf(stderr, "%s: illegal option -- %c\n",
			    p, eoptopt);
		}
		return (BADCH);
	}
	if (delim == (int)'+') {
		/*
		 * '+' is only allowed with numbers
		 */
		if (!*place)
			++eoptind;
		if (eopterr) {
			if (!(p = strrchr(*nargv, '/')))
				p = *nargv;
			else
				++p;
			(void)fprintf(stderr,
				"%s: illegal '+' delimiter with option -- %c\n",
				p, eoptopt);
		}
		return (BADCH);
	}
	++oli;
	if ((*oli != ':') && (*oli != '?')) {
		/*
		 * don't need argument
		 */
		eoptarg = NULL;
		if (!*place)
			++eoptind;
		return (eoptopt);
	}

	if (*place) {
		/*
		 * no white space
		 */
		eoptarg = place;
	} else if (*oli == '?') {
		/*
		 * no arg, but NOT required
		 */
		eoptarg = NULL;
	} else if (nargc <= ++eoptind) {
		/*
		 * no arg, but IS required
		 */
		place = emsg;
		if (eopterr) {
			if (!(p = strrchr(*nargv, '/')))
				p = *nargv;
			else
				++p;
			(void)fprintf(stderr,
			    "%s: option requires an argument -- %c\n", p,
			    eoptopt);
		}
		return (BADCH);
	} else {
		/*
		 * arg has white space
		 */
		eoptarg = nargv[eoptind];
	}
	place = emsg;
	++eoptind;
	return (eoptopt);
}
