/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
/*  searchp  --  search through pathlist for file
 *
 *  Usage:  p = searchp (path,file,fullname,func);
 *	char *p, *path, *file, *fullname;
 *	int (*func)();
 *
 *  Searchp will parse "path", a list of pathnames separated
 *  by colons, prepending each pathname to "file".  The resulting
 *  filename will be passed to "func", a function provided by the
 *  user.  This function must return zero if the search is
 *  successful (i.e. ended), and non-zero if the search must
 *  continue.  If the function returns zero (success), then
 *  searching stops, the full filename is placed into "fullname",
 *  and searchp returns 0.  If the pathnames are all unsuccessfully
 *  examined, then searchp returns -1.
 *  If "file" begins with a slash, it is assumed to be an
 *  absolute pathname and the "path" list is not used.  Note
 *  that this rule is used by Bell's cc also; whereas Bell's
 *  sh uses the rule that any filename which CONTAINS a slash
 *  is assumed to be absolute.  The execlp and execvp procedures
 *  also use this latter rule.  In my opinion, this is bogosity.
 *
 *  HISTORY
 * 01-Apr-86  Rudy Nedved (ern) at Carnegie-Mellon University
 *	4.1BSD system ignores trailing slashes. 4.2BSD does not.
 *	Therefore don't add a seperating slash if there is a null
 *	filename.
 *
 * 23-Oct-82  Steven Shafer (sas) at Carnegie-Mellon University
 *	Fixed two bugs: (1) calling function as "func" instead of
 *	"(*func)", (2) omitting trailing null name implied by trailing
 *	colon in path.  Latter bug fixed by introducing "lastchar" and
 *	changing final loop test to look for "*lastchar" instead of
 *	"*nextpath".
 *
 * 20-Nov-79  Steven Shafer (sas) at Carnegie-Mellon University
 *	Created for VAX.  If you're thinking of using this, you probably
 *	should look at openp() and fopenp() (or the "want..." routines)
 *	instead.
 *
 */
#include "config.h"

int
searchp(const char *spath, char *file, char *fullname, int (*func)(char *))
{
	const char *nextpath, *nextchar, *lastchar;
	char *fname;
	int failure;

	nextpath = ((*file == '/') ? "" : spath);
	do {
		fname = fullname;
		nextchar = nextpath;
		while (*nextchar && (*nextchar != ':')) {
			*fname++ = *nextchar++;
		}
		if (nextchar != nextpath && *file) {
			*fname++ = '/';
		}
		lastchar = nextchar;
		nextpath = ((*nextchar) ? nextchar + 1 : nextchar);
		nextchar = file;        /* append file */
		while (*nextchar) {
			*fname++ = *nextchar++;
		}
		*fname = '\0';
		failure = (*func)(fullname);
	}while (failure && (*lastchar));
	return failure ? -1 : 0;
}

