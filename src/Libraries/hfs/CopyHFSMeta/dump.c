/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include <ctype.h>
#include <unistd.h>

#define MIN(a, b) \
	({ __typeof(a) _a = (a); __typeof(b) _b = (b);	\
		(_a < _b) ? _a : _b; })

enum { WIDTH = 16, };

/*
 * Debug functions only.
 */
void
DumpData(const void *data, size_t len)
{
	unsigned char *base = (unsigned char*)data;
	unsigned char *end = base + len;
	unsigned char *cp = base;
	int allzeroes = 0;

	while (cp < end) {
		unsigned char *tend = MIN(end, cp + WIDTH);
		unsigned char *tmp;
		int i;
		size_t gap = (cp + WIDTH) - tend;

		if (gap != 0 || tend == end)
			allzeroes = 0;
		if (allzeroes) {
			for (tmp = cp; tmp < tend; tmp++) {
				if (*tmp) {
					allzeroes = 0;
					break;
				}
			}
			if (allzeroes == 1) {
				printf(". . .\n");
				allzeroes = 2;
			}
			if (allzeroes) {
				cp += WIDTH;
				continue;
			}
		}
		allzeroes = 1;

		printf("%04x:  ", (int)(cp - base));
		for (i = 0, tmp = cp; tmp < tend; tmp++) {
			printf("%02x", *tmp);
			if (++i % 2 == 0)
				printf(" ");
			if (*tmp)
				allzeroes = 0;
		}
		for (i = (int)gap; i >= 0; i--) {
			printf("  ");
			if (i % 2 == 1)
				printf(" ");
		}
		printf("    |");
		for (tmp = cp; tmp < tend; tmp++) {
			printf("%c", isalnum(*tmp) ? *tmp : '.');
		}
		for (i = 0; i < gap; i++) {
			printf(" ");
		}
		printf("|\n");
		cp += WIDTH;
	}

	return;

}
