/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include "ftp_locl.h"
RCSID("$Id$");

void
domacro(int argc, char **argv)
{
	int i, j, count = 2, loopflg = 0;
	char *cp1, *cp2, line2[200];
	struct cmd *c;

	if (argc < 2 && !another(&argc, &argv, "macro name")) {
		printf("Usage: %s macro_name.\n", argv[0]);
		code = -1;
		return;
	}
	for (i = 0; i < macnum; ++i) {
		if (!strncmp(argv[1], macros[i].mac_name, 9)) {
			break;
		}
	}
	if (i == macnum) {
		printf("'%s' macro not found.\n", argv[1]);
		code = -1;
		return;
	}
	strlcpy(line2, line, sizeof(line2));
TOP:
	cp1 = macros[i].mac_start;
	while (cp1 != macros[i].mac_end) {
		while (isspace((unsigned char)*cp1)) {
			cp1++;
		}
		cp2 = line;
		while (*cp1 != '\0') {
		      size_t len;
		      switch(*cp1) {
		   	    case '\\':
				if (line + sizeof(line) - 2 < cp2)
				    goto out;
				*cp2++ = *++cp1;
				 break;
			    case '$':
				 if (isdigit((unsigned char)*(cp1+1))) {
				    j = 0;
				    while (isdigit((unsigned char)*++cp1)) {
					  j = 10*j +  *cp1 - '0';
				    }
				    cp1--;
				    if (argc - 2 >= j) {
					len = sizeof(line) - (cp2 - line) - 1;
					if (strlcpy(cp2, argv[j+1], len) >= len)
					    goto out;
					cp2 += strlen(argv[j+1]);
				    }
				    break;
				 }
				 if (*(cp1+1) == 'i') {
					loopflg = 1;
					cp1++;
					if (count < argc) {
					   len = sizeof(line) - (cp2 - line) - 1;
					   if (strlcpy(cp2, argv[count], len) >= len)
					       goto out;
					   cp2 += strlen(argv[count]);
					}
					break;
				}
				/* intentional drop through */
			    default:
				if (line + sizeof(line) - 2 < cp2)
				    goto out;
				*cp2++ = *cp1;
				break;
		      }
		      if (*cp1 != '\0') {
			 cp1++;
		      }
		}
	out:
		*cp2 = '\0';
		makeargv();
		c = getcmd(margv[0]);
		if (c == (struct cmd *)-1) {
			printf("?Ambiguous command\n");
			code = -1;
		}
		else if (c == 0) {
			printf("?Invalid command\n");
			code = -1;
		}
		else if (c->c_conn && !connected) {
			printf("Not connected.\n");
			code = -1;
		}
		else {
			if (verbose) {
				printf("%s\n",line);
			}
			(*c->c_handler)(margc, margv);
			if (bell && c->c_bell) {
				putchar('\007');
			}
			strlcpy(line, line2, sizeof(line));
			makeargv();
			argc = margc;
			argv = margv;
		}
		if (cp1 != macros[i].mac_end) {
			cp1++;
		}
	}
	if (loopflg && ++count < argc) {
		goto TOP;
	}
}
