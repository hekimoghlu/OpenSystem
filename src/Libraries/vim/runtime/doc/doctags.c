/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define LINELEN 200

	int
main(int argc, char **argv)
{
	char	line[LINELEN];
	char	*p1, *p2;
	char	*p;
	FILE	*fd;
	int		len;
	int		in_example;

	if (argc <= 1)
	{
		fprintf(stderr, "Usage: doctags docfile ... >tags\n");
		exit(1);
	}
	printf("help-tags\ttags\t1\n");
	while (--argc > 0)
	{
		++argv;
		fd = fopen(argv[0], "r");
		if (fd == NULL)
		{
			fprintf(stderr, "Unable to open %s for reading\n", argv[0]);
			continue;
		}
		in_example = 0;
		while (fgets(line, LINELEN, fd) != NULL)
		{
			if (in_example)
			{
				// skip over example; non-blank in first column ends example
				if (strchr(" \t\n\r", line[0]) != NULL)
					continue;
				in_example = 0;
			}
			p1 = strchr(line, '*');				// find first '*'
			while (p1 != NULL)
			{
				p2 = strchr(p1 + 1, '*');		// find second '*'
				if (p2 != NULL && p2 > p1 + 1)	// skip "*" and "**"
				{
					for (p = p1 + 1; p < p2; ++p)
						if (*p == ' ' || *p == '\t' || *p == '|')
							break;
					// Only accept a *tag* when it consists of valid
					// characters, there is white space before it and is
					// followed by a white character or end-of-line.
					if (p == p2
							&& (p1 == line || p1[-1] == ' ' || p1[-1] == '\t')
								&& (strchr(" \t\n\r", p[1]) != NULL
									|| p[1] == '\0'))
					{
						*p2 = '\0';
						++p1;
						printf("%s\t%s\t/*", p1, argv[0]);
						while (*p1)
						{
							// insert backslash before '\\' and '/'
							if (*p1 == '\\' || *p1 == '/')
								putchar('\\');
							putchar(*p1);
							++p1;
						}
						printf("*\n");
						p2 = strchr(p2 + 1, '*');		// find next '*'
					}
				}
				p1 = p2;
			}
			len = strlen(line);
			if ((len == 2 && strcmp(&line[len - 2], ">\n") == 0)
					|| (len >= 3 && strcmp(&line[len - 3], " >\n") == 0))
				in_example = 1;
		}
		fclose(fd);
	}
	return 0;
}
