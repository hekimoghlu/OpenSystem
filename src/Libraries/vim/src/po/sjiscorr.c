/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

	int
main(int argc, char **argv)
{
	char buffer[BUFSIZ];
	char *p;

	while (fgets(buffer, BUFSIZ, stdin) != NULL)
	{
		for (p = buffer; *p != 0; p++)
		{
			if (strncmp(p, "charset=utf-8", 13) == 0
				|| strncmp(p, "charset=UTF-8", 13) == 0)
			{
				fputs("charset=CP932", stdout);
				p += 12;
			}
			else if (strncmp(p, "# Original translations", 23) == 0)
			{
				fputs("# Generated from ja.po, DO NOT EDIT.", stdout);
				while (p[1] != '\n')
					++p;
			}
			else if (*(unsigned char *)p == 0x81 && p[1] == '_')
			{
				putchar('\\');
				++p;
			}
			else
			{
				if (*p & 0x80)
				{
					putchar(*p++);
					if (*p == '\\')
						putchar(*p);
				}
				putchar(*p);
			}
		}
	}
}
