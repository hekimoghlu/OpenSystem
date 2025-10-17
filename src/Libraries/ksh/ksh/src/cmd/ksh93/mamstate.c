/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
 * AT&T Bell Laboratories
 * make abstract machine file state support
 *
 * mamstate reference [ file ... | <files ]
 *
 * stdout is list of <file,delta> pairs where delta
 * is diff between reference and file times
 * non-existent files are not listed
 */

#if !lint
static char id[] = "\n@(#)$Id: mamstate (AT&T Bell Laboratories) 1989-06-26 $\0\n";
#endif

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

main(argc, argv)
int		argc;
register char**	argv;
{
	register char*	s;
	register int	c;
	long		ref;
	struct stat	st;
	char		buf[1024];

	if (!(s = *++argv) || stat(s, &st))
	{
		fprintf(stderr, "Usage: mamstate reference [ file ... | <files ]\n");
		exit(1);
	}
	ref = (long)st.st_mtime;
	if (s = *++argv) do
	{
		if (!stat(s, &st))
			printf("%s %ld\n", s, (long)st.st_mtime - ref);
	} while (s = *++argv);
	else do
	{
		s = buf;
		while ((c = getchar()) != EOF && c != ' ' && c != '\n')
			if (s < buf + sizeof(buf) - 1) *s++ = c;
		if (s > buf)
		{
			*s = 0;
			if (!stat(buf, &st))
				printf("%s %ld\n", buf, (long)st.st_mtime - ref);
		}
	} while (c != EOF);
	exit(0);
}
