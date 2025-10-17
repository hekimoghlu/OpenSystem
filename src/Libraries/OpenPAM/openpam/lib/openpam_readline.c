/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <security/pam_appl.h>
#include "openpam_impl.h"

#define MIN_LINE_LENGTH 128

/*
 * OpenPAM extension
 *
 * Read a line from a file.
 */

char *
openpam_readline(FILE *f, int *lineno, size_t *lenp)
{
	char *line;
	size_t len, size;
	int ch;

	if ((line = malloc(MIN_LINE_LENGTH)) == NULL)
		return (NULL);
	size = MIN_LINE_LENGTH;
	len = 0;

#define line_putch(ch) do { \
	if (len >= size - 1) { \
		char *tmp = realloc(line, size *= 2); \
		if (tmp == NULL) \
			goto fail; \
		line = tmp; \
	} \
	line[len++] = ch; \
	line[len] = '\0'; \
} while (0)

	for (;;) {
		ch = fgetc(f);
		/* strip comment */
		if (ch == '#') {
			do {
				ch = fgetc(f);
			} while (ch != EOF && ch != '\n');
		}
		/* eof */
		if (ch == EOF) {
			/* remove trailing whitespace */
			while (len > 0 && isspace((int)line[len - 1]))
				--len;
			line[len] = '\0';
			if (len == 0)
				goto fail;
			break;
		}
		/* eol */
		if (ch == '\n') {
			if (lineno != NULL)
				++*lineno;

			/* remove trailing whitespace */
			while (len > 0 && isspace((int)line[len - 1]))
				--len;
			line[len] = '\0';
			/* skip blank lines */
			if (len == 0)
				continue;
			/* continuation */
			if (line[len - 1] == '\\') {
				line[--len] = '\0';
				/* fall through to whitespace case */
			} else {
				break;
			}
		}
		/* whitespace */
		if (isspace(ch)) {
			/* ignore leading whitespace */
			/* collapse linear whitespace */
			if (len > 0 && line[len - 1] != ' ')
				line_putch(' ');
			continue;
		}
		/* anything else */
		line_putch(ch);
	}

	if (lenp != NULL)
		*lenp = len;
	return (line);
 fail:
	FREE(line);
	return (NULL);
}

/**
 * The =openpam_readline function reads a line from a file, and returns it
 * in a NUL-terminated buffer allocated with =malloc.
 *
 * The =openpam_readline function performs a certain amount of processing
 * on the data it reads.
 * Comments (introduced by a hash sign) are stripped, as is leading and
 * trailing whitespace.
 * Any amount of linear whitespace is collapsed to a single space.
 * Blank lines are ignored.
 * If a line ends in a backslash, the backslash is stripped and the next
 * line is appended.
 *
 * If =lineno is not =NULL, the integer variable it points to is
 * incremented every time a newline character is read.
 *
 * If =lenp is not =NULL, the length of the line (not including the
 * terminating NUL character) is stored in the variable it points to.
 *
 * The caller is responsible for releasing the returned buffer by passing
 * it to =free.
 */
