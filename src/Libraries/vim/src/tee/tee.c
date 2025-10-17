/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef _MSC_VER
# include <unistd.h>
#endif
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#ifdef _WIN32
# define sysconf(x) -1
#endif

void usage(void)
{
	fprintf(stderr,
"tee usage:\n\
\ttee [-a] file ... file_n\n\
\n\
\t-a\tappend to files instead of truncating\n\
\nTee reads its input, and writes to each of the specified files,\n\
as well as to the standard output.\n\
\n\
This version supplied with Vim 4.2 to make ':make' possible.\n\
For a more complete and stable version, consider getting\n\
[a port of] the GNU shellutils package.\n\
");
}

/*
 * fread only returns when count is read or at EOF.
 * We could use fgets, but I want to be able to handle binary blubber.
 */

int
myfread(char *buf, int elsize /*ignored*/, int max, FILE *fp)
{
	int	c;
	int	n = 0;

	while ((n < max) && ((c = getchar()) != EOF))
	{
		*(buf++) = c;
		n++;
		if (c == '\n' || c == '\r')
			break;
	}
	return n;
}


int
main(int argc, char *argv[])
{
	int	append = 0;
	size_t	numfiles;
	int	maxfiles;
	FILE	**filepointers;
	int	i;
	char	buf[BUFSIZ];
	int	n;
	int	optind = 1;

	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-')
			break;
		if (!strcmp(argv[i], "-a"))
			append++;
		else
			usage();
		optind++;
	}

	numfiles = argc - optind;

	if (numfiles == 0)
	{
		fprintf(stderr, "doesn't make much sense using tee without any file name arguments...\n");
		usage();
		exit(2);
	}

	maxfiles = sysconf(_SC_OPEN_MAX);	/* or fill in 10 or so */
	if (maxfiles < 0)
		maxfiles = 10;
	if (numfiles + 3 > maxfiles)	/* +3 accounts for stdin, out, err */
	{
		fprintf(stderr, "Sorry, there is a limit of max %d files.\n", maxfiles - 3);
		exit(1);
	}
	filepointers = calloc(numfiles, sizeof(FILE *));
	if (filepointers == NULL)
	{
		fprintf(stderr, "Error allocating memory for %ld files\n",
															   (long)numfiles);
		exit(1);
	}
	for (i = 0; i < numfiles; i++)
	{
		filepointers[i] = fopen(argv[i+optind], append ? "ab" : "wb");
		if (filepointers[i] == NULL)
		{
			fprintf(stderr, "Can't open \"%s\"\n", argv[i+optind]);
			exit(1);
		}
	}
#ifdef _WIN32
	setmode(fileno(stdin),  O_BINARY);
	fflush(stdout);	/* needed for _fsetmode(stdout) */
	setmode(fileno(stdout),  O_BINARY);
#endif

	while ((n = myfread(buf, sizeof(char), sizeof(buf), stdin)) > 0)
	{
		fwrite(buf, sizeof(char), n, stdout);
		fflush(stdout);
		for (i = 0; i < numfiles; i++)
		{
			if (filepointers[i] &&
			     fwrite(buf, sizeof(char), n, filepointers[i]) != n)
			{
				fprintf(stderr, "Error writing to file \"%s\"\n", argv[i+optind]);
				fclose(filepointers[i]);
				filepointers[i] = NULL;
			}
		}
	}
	for (i = 0; i < numfiles; i++)
	{
		if (filepointers[i])
			fclose(filepointers[i]);
	}

	exit(0);
}
