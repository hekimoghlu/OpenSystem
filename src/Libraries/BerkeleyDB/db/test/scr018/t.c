/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include <sys/types.h>

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
chk(f)
	char *f;
{
	int ch, l, r;

	if (freopen(f, "r", stdin) == NULL) {
		fprintf(stderr, "%s: %s\n", f, strerror(errno));
		exit(EXIT_FAILURE);
	}
	for (l = 1, r = 0; (ch = getchar()) != EOF;) {
		if (ch != ',')
			goto next;
		do { ch = getchar(); } while (isblank(ch));
		if (ch != '\n')
			goto next;
		++l;
		do { ch = getchar(); } while (isblank(ch));
		if (ch != '}')
			goto next;
		r = 1;
		printf("%s: line %d\n", f, l);

next:		if (ch == '\n')
			++l;
	}
	return (r);
}

int
main(int argc, char *argv[])
{
	int r;

	for (r = 0; *++argv != NULL;)
		if (chk(*argv))
			r = 1;
	return (r);
}
