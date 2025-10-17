/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
 * Parse CSV object serialization format (RFC-4180, RFC-7111)
 */

#ifndef TEST
#include "file.h"

#ifndef lint
FILE_RCSID("@(#)$File: is_csv.c,v 1.6 2020/08/09 16:43:36 christos Exp $")
#endif

#include <string.h>
#include "magic.h"
#else
#include <sys/types.h>
#endif


#ifdef DEBUG
#include <stdio.h>
#define DPRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DPRINTF(fmt, ...)
#endif

/*
 * if CSV_LINES == 0:
 *	check all the lines in the buffer
 * otherwise:
 *	check only up-to the number of lines specified
 *
 * the last line count is always ignored if it does not end in CRLF
 */
#ifndef CSV_LINES
#define CSV_LINES 10
#endif

static int csv_parse(const unsigned char *, const unsigned char *);

static const unsigned char *
eatquote(const unsigned char *uc, const unsigned char *ue)
{
	int quote = 0;

	while (uc < ue) {
		unsigned char c = *uc++;
		if (c != '"') {
			// We already got one, done.
			if (quote) {
				return --uc;
			}
			continue;
		}
		if (quote) {
			// quote-quote escapes
			quote = 0;
			continue;
		}
		// first quote
		quote = 1;
	}
	return ue;
}

static int
csv_parse(const unsigned char *uc, const unsigned char *ue)
{
	size_t nf = 0, tf = 0, nl = 0;

	while (uc < ue) {
		switch (*uc++) {
		case '"':
			// Eat until the matching quote
			uc = eatquote(uc, ue);
			break;
		case ',':
			nf++;
			break;
		case '\n':
			DPRINTF("%zu %zu %zu\n", nl, nf, tf);
			nl++;
#if CSV_LINES
			if (nl == CSV_LINES)
				return tf != 0 && tf == nf;
#endif
			if (tf == 0) {
				// First time and no fields, give up
				if (nf == 0) 
					return 0;
				// First time, set the number of fields
				tf = nf;
			} else if (tf != nf) {
				// Field number mismatch, we are done.
				return 0;
			}
			nf = 0;
			break;
		default:
			break;
		}
	}
	return tf && nl > 2;
}

#ifndef TEST
int
file_is_csv(struct magic_set *ms, const struct buffer *b, int looks_text)
{
	const unsigned char *uc = CAST(const unsigned char *, b->fbuf);
	const unsigned char *ue = uc + b->flen;
	int mime = ms->flags & MAGIC_MIME;

	if (!looks_text)
		return 0;

	if ((ms->flags & (MAGIC_APPLE|MAGIC_EXTENSION)) != 0)
		return 0;

	if (!csv_parse(uc, ue))
		return 0;

	if (mime == MAGIC_MIME_ENCODING)
		return 1;

	if (mime) {
		if (file_printf(ms, "text/csv") == -1)
			return -1;
		return 1;
	}

	if (file_printf(ms, "CSV text") == -1)
		return -1;

	return 1;
}

#else

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <err.h>

int
main(int argc, char *argv[])
{
	int fd, rv;
	struct stat st;
	unsigned char *p;

	if ((fd = open(argv[1], O_RDONLY)) == -1)
		err(EXIT_FAILURE, "Can't open `%s'", argv[1]);

	if (fstat(fd, &st) == -1)
		err(EXIT_FAILURE, "Can't stat `%s'", argv[1]);

	if ((p = malloc(st.st_size)) == NULL)
		err(EXIT_FAILURE, "Can't allocate %jd bytes",
		    (intmax_t)st.st_size);
	if (read(fd, p, st.st_size) != st.st_size)
		err(EXIT_FAILURE, "Can't read %jd bytes",
		    (intmax_t)st.st_size);
	printf("is csv %d\n", csv_parse(p, p + st.st_size));
	return 0;
}
#endif
