/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * rmdir
 */

static const char usage[] =
"[-?\n@(#)$Id: rmdir (AT&T Research) 2006-08-24 $\n]"
USAGE_LICENSE
"[+NAME?rmdir - remove empty directories]"
"[+DESCRIPTION?\brmdir\b deletes each given directory.  The directory "
	"must be empty; containing no entries other than \b.\b or \b..\b.  "
	"If a directory and a subdirectory of that directory are specified "
	"as operands, the subdirectory must be specified before the parent "
	"so that the parent directory will be empty when \brmdir\b attempts "
	"to remove it.]"
"[e:ignore-fail-on-non-empty?Ignore each non-empty directory failure.]"
"[p:parents?Remove each explicit \adirectory\a argument directory that "
	"becomes empty after its child directories are removed.]"
"[s:suppress?Suppress the message printed on the standard error when "
	"\b-p\b is in effect.]"
"\n"
"\ndirectory ...\n"
"\n"
"[+EXIT STATUS?]{"
        "[+0?All directories deleted successfully.]"
        "[+>0?One or more directories could not be deleted.]"
"}"
"[+SEE ALSO?\bmkdir\b(1), \brm\b(1), \brmdir\b(2), \bunlink\b(2)]"
;

#include <cmd.h>

int
b_rmdir(int argc, char** argv, Shbltin_t* context)
{
	register char*	dir;
	register char*	end;
	register int	n;
	int		eflag = 0;
	int		pflag = 0;
	int		sflag = 0;

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case 'e':
			eflag = 1;
			continue;
		case 'p':
			pflag = 1;
			continue;
		case 's':
			sflag = 1;
			continue;
		case ':':
			error(2, "%s", opt_info.arg);
			break;
		case '?':
			error(ERROR_usage(2), "%s", opt_info.arg);
			break;
		}
		break;
	}
	argv += opt_info.index;
	if (error_info.errors || !*argv)
		error(ERROR_usage(2), "%s", optusage(NiL));
	if (!pflag)
		sflag = 0;
	while (dir = *argv++)
	{
		end = dir;
		if (pflag) end += strlen(dir);
		n = 0;
		for (;;)
		{
			if (rmdir(dir) < 0)
			{
				if (!eflag || errno != EEXIST
#ifdef ENOTEMPTY
				    && errno != ENOTEMPTY
#endif
				    )
				{
					if (sflag)
						error_info.errors++;
					else
						error(ERROR_system(0), "%s: cannot remove", dir);
				}
				break;
			}
			if (n) *end = '/';
			else n = 1;
			do if (end <= dir) goto next; while (*--end != '/');
			do if (end <= dir) goto next; while (*(end - 1) == '/' && end--);
			*end = 0;
		}
	next:	;
	}
	return(error_info.errors != 0);
}

