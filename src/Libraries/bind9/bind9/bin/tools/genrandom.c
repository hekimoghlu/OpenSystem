/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
/* $Id: genrandom.c,v 1.7 2010/05/17 23:51:04 tbox Exp $ */

/*! \file */
#include <config.h>

#include <isc/commandline.h>
#include <isc/print.h>
#include <isc/stdlib.h>
#include <isc/util.h>

#include <stdio.h>
#include <string.h>

const char *program = "genrandom";

ISC_PLATFORM_NORETURN_PRE static void
usage(void) ISC_PLATFORM_NORETURN_POST;

static void
usage(void) {
	fprintf(stderr, "usage: %s [-n 2..9] k file\n", program);
	exit(1);
}

static void
generate(char *filename, unsigned int bytes) {
	FILE *fp;

	fp = fopen(filename, "w");
	if (fp == NULL) {
		printf("failed to open %s\n", filename);
		exit(1);
	}

	while (bytes > 0) {
#ifndef HAVE_ARC4RANDOM
		unsigned short int x = (rand() & 0xFFFF);
#else
		unsigned short int x = (arc4random() & 0xFFFF);
#endif
		unsigned char c = x & 0xFF;
		if (putc(c, fp) == EOF) {
			printf("error writing to %s\n", filename);
			exit(1);
		}
		c = x >> 8;
		if (putc(c, fp) == EOF) {
			printf("error writing to %s\n", filename);
			exit(1);
		}
		bytes -= 2;
	}
	fclose(fp);
}

int
main(int argc, char **argv) {
	unsigned int bytes;
	unsigned int k;
	char *endp;
	int c, i, n = 1;
	size_t len;
	char *name;

	isc_commandline_errprint = ISC_FALSE;

	while ((c = isc_commandline_parse(argc, argv, "hn:")) != EOF) {
		switch (c) {
		case 'n':
			n = strtol(isc_commandline_argument, &endp, 10);
			if ((*endp != 0) || (n <= 1) || (n > 9))
				usage();
			break;

		case '?':
			if (isc_commandline_option != '?')
				fprintf(stderr, "%s: invalid argument -%c\n",
					program, isc_commandline_option);
			/* FALLTHROUGH */
		case 'h':
			usage();

		default:
			fprintf(stderr, "%s: unhandled option -%c\n",
				program, isc_commandline_option);
			exit(1);
		}
	}

	if (isc_commandline_index + 2 != argc)
		usage();

	k = strtoul(argv[isc_commandline_index++], &endp, 10);
	if (*endp != 0)
		usage();
	bytes = k << 10;

#ifndef HAVE_ARC4RANDOM
	srand(0x12345678);
#endif
	if (n == 1) {
		generate(argv[isc_commandline_index], bytes);
		return (0);
	}

	len = strlen(argv[isc_commandline_index]);
	INSIST((len + 2) > len);
	len += 2;
	name = (char *) malloc(len);
	if (name == NULL) {
		perror("malloc");
		exit(1);
	}

	for (i = 1; i <= n; i++) {
		snprintf(name, len, "%s%d", argv[isc_commandline_index], i);
		generate(name, bytes);
	}
	free(name);

	return (0);
}
