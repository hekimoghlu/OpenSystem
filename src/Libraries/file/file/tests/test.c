/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "magic.h"

static const char *prog;

static void *
xrealloc(void *p, size_t n)
{
	p = realloc(p, n);
	if (p == NULL) {
		(void)fprintf(stderr, "%s ERROR slurping file: %s\n",
			prog, strerror(errno));
		exit(10);
	}
	return p;
}

static char *
slurp(FILE *fp, size_t *final_len)
{
	size_t len = 256;
	int c;
	char *l = (char *)xrealloc(NULL, len), *s = l;

	for (c = getc(fp); c != EOF; c = getc(fp)) {
		if (s == l + len) {
			l = xrealloc(l, len * 2);
			s = l + len;
			len *= 2;
		}
		*s++ = c;
	}
	if (s == l + len) {
		l = (char *)xrealloc(l, len + 1);
		s = l + len;
	}
	*s++ = '\0';

	*final_len = s - l;
	l = (char *)xrealloc(l, s - l);
	return l;
}

int
main(int argc, char **argv)
{
	struct magic_set *ms;
	const char *result;
	size_t result_len, desired_len;
	char *desired = NULL;
	int e = EXIT_FAILURE;
	FILE *fp;


	prog = strrchr(argv[0], '/');
	if (prog)
		prog++;
	else
		prog = argv[0];

	ms = magic_open(MAGIC_NONE);
	if (ms == NULL) {
		(void)fprintf(stderr, "%s: ERROR opening MAGIC_NONE: %s\n",
		    prog, strerror(errno));
		return e;
	}
	if (magic_load(ms, NULL) == -1) {
		(void)fprintf(stderr, "%s: ERROR loading with NULL file: %s\n",
		    prog, magic_error(ms));
		goto bad;
	}

	if (argc == 1) {
		e = 0;
		goto bad;
	}

	if (argc != 3) {
		(void)fprintf(stderr, "Usage: %s TEST-FILE RESULT\n", prog);
		magic_close(ms);
		goto bad;
	}
	if ((result = magic_file(ms, argv[1])) == NULL) {
		(void)fprintf(stderr, "%s: ERROR loading file %s: %s\n",
		    prog, argv[1], magic_error(ms));
		goto bad;
	}
	fp = fopen(argv[2], "r");
	if (fp == NULL) {
		(void)fprintf(stderr, "%s: ERROR opening `%s': %s",
		    prog, argv[2], strerror(errno));
		goto bad;
	}
	desired = slurp(fp, &desired_len);
	fclose(fp);
	(void)printf("%s: %s\n", argv[1], result);
	if (strcmp(result, desired) != 0) {
	    result_len = strlen(result);
	    (void)fprintf(stderr, "%s: ERROR: result was (len %zu)\n%s\n"
		"expected (len %zu)\n%s\n", prog, result_len, result,
		desired_len, desired);
	    goto bad;
	}
	e = 0;
bad:
	free(desired);
	magic_close(ms);
	return e;
}
