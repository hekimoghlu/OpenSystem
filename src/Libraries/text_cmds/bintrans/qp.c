/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern int main_quotedprintable(int, char *[]);

static int
hexval(int c)
{
	if ('0' <= c && c <= '9')
		return c - '0';
	return (10 + c - 'A');
}


static int
decode_char(const char *s)
{
	return (16 * hexval(toupper(s[1])) + hexval(toupper(s[2])));
}


static void
decode_quoted_printable(const char *body, FILE *fpo)
{
	while (*body != '\0') {
		switch (*body) {
		case '=':
			if (strlen(body) < 2) {
				fputc(*body, fpo);
				break;
			}

			if (body[1] == '\r' && body[2] == '\n') {
				body += 2;
				break;
			}
			if (body[1] == '\n') {
				body++;
				break;
			}
			if (strchr("0123456789ABCDEFabcdef", body[1]) == NULL) {
				fputc(*body, fpo);
				break;
			}
			if (strchr("0123456789ABCDEFabcdef", body[2]) == NULL) {
				fputc(*body, fpo);
				break;
			}
			fputc(decode_char(body), fpo);
			body += 2;
			break;
		default:
			fputc(*body, fpo);
			break;
		}
		body++;
	}
}

static void
encode_quoted_printable(const char *body, FILE *fpo)
{
	const char *end = body + strlen(body);
	size_t linelen = 0;
	char prev = '\0';

	while (*body != '\0') {
		if (linelen == 75) {
			fputs("=\r\n", fpo);
			linelen = 0;
		}
		if (!isascii(*body) ||
		    *body == '=' ||
		    (*body == '.' && body + 1 < end &&
		      (body[1] == '\n' || body[1] == '\r'))) {
			fprintf(fpo, "=%02X", (unsigned char)*body);
			linelen += 2;
			prev = *body;
		} else if (*body < 33 && *body != '\n') {
			if ((*body == ' ' || *body == '\t') &&
			    body + 1 < end &&
			    (body[1] != '\n' && body[1] != '\r')) {
				fputc(*body, fpo);
				prev = *body;
			} else {
				fprintf(fpo, "=%02X", (unsigned char)*body);
				linelen += 2;
				prev = '_';
			}
		} else if (*body == '\n') {
			if (prev == ' ' || prev == '\t') {
				fputc('=', fpo);
			}
			fputc('\n', fpo);
			linelen = 0;
			prev = 0;
		} else {
			fputc(*body, fpo);
			prev = *body;
		}
		body++;
		linelen++;
	}
}

static void
qp(FILE *fp, FILE *fpo, bool encode)
{
	char *line = NULL;
	size_t linecap = 0;
	void (*codec)(const char *line, FILE *f);

	codec = encode ? encode_quoted_printable : decode_quoted_printable ;

	while (getline(&line, &linecap, fp) > 0)
		codec(line, fpo);
	free(line);
}

static void
usage(void)
{
	fprintf(stderr,
	   "usage: bintrans qp [-u] [-o outputfile] [file name]\n");
}

int
main_quotedprintable(int argc, char *argv[])
{
	int i;
	bool encode = true;
	FILE *fp = stdin;
	FILE *fpo = stdout;

	for (i = 1; i < argc; ++i) {
		if (argv[i][0] == '-') {
			switch (argv[i][1]) {
			case 'o':
				if (++i >= argc) {
					fprintf(stderr, "qp: -o requires a file name.\n");
					exit(EXIT_FAILURE);
				}
				fpo = fopen(argv[i], "w");
				if (fpo == NULL) {
					perror(argv[i]);
					exit(EXIT_FAILURE);
				}
				break;
			case 'u':
				encode = false;
				break;
			default:
				usage();
				exit(EXIT_FAILURE);
			}
		} else {
			fp = fopen(argv[i], "r");
			if (fp == NULL) {
				perror(argv[i]);
				exit(EXIT_FAILURE);
			}
		}
	}
	qp(fp, fpo, encode);

	return (EXIT_SUCCESS);
}
