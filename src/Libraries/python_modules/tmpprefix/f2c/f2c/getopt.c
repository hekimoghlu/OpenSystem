/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
/* Source for a "getopt" command, as invoked by the "fc" script. */

#include <stdio.h>

static char opts[256];	/* assume 8-bit bytes */

 int
#ifdef KR_headers
main(argc, argv) int argc; char **argv;
#else
main(int argc, char **argv)
#endif
{
	char **av, *fmt, *s, *s0;
	int i;

	if (argc < 2) {
		fprintf(stderr, "Usage: getopt optstring arg1 arg2...\n");
		return 1;
		}
	for(s = argv[1]; *s; ) {
		i = *(unsigned char *)s++;
		if (!opts[i])
			opts[i] = 1;
		if (*s == ':') {
			s++;
			opts[i] = 2;
			}
		}
	/* scan for legal args */
	av = argv + 2;
 nextarg:
	while(s = *av++) {
		if (*s++ != '-' || s[0] == '-' && s[1] == 0)
			break;
		while(i = *(unsigned char *)s++) {
			switch(opts[i]) {
			  case 0:
				fprintf(stderr,
					"getopt: Illegal option -- %c\n", s[-1]);
				return 1;
			  case 2:
				s0 = s - 1;
				if (*s || *av++)
					goto nextarg;
				fprintf(stderr,
				 "getopt: Option requires an argument -- %c\n",
					*s0);
				return 1;
			  }
			}
		}
	/* output modified args */
	av = argv + 2;
	fmt = "-%c";
 nextarg1:
	while(s = *av++) {
		if (s[0] != '-')
			break;
		if (*++s == '-' && !s[1]) {
			s = *av++;
			break;
			}
		while(*s) {
			printf(fmt, *s);
			fmt = " -%c";
			if (opts[*(unsigned char *)s++] == 2) {
				if (!*s)
					s = *av++;
				printf(" %s", s);
				goto nextarg1;
				}
			}
		}
	printf(*fmt == ' ' ? " --" : "--");
	for(; s; s = *av++)
		printf(" %s", s);
	printf("\n");
	return 0;
	}
