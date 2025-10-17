/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#include <sys/param.h>
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>

#ifdef __APPLE__
#include <err.h>
#include "bintrans.h"
#endif

extern int	main_decode(int, char *[]);
extern int	main_encode(int, char *[]);
#ifdef __APPLE__
extern int	main_base64_decode(const char *, const char *);
extern int	main_base64_encode(const char *, const char *, const char *);
#else
extern int	main_base64_decode(const char *);
extern int	main_base64_encode(const char *, const char *);
#endif
extern int	main_quotedprintable(int, char*[]);

static int	search(const char *const);
static void	usage_base64(bool);
static void	version_base64(void);
static void	base64_encode_or_decode(int, char *[]);

enum coders {
	uuencode, uudecode, b64encode, b64decode, base64, qp
};

int
main(int argc, char *argv[])
{
	const char *const progname = getprogname();
	int coder = search(progname);

	if (coder == -1 && argc > 1) {
		argc--;
		argv++;
		coder = search(argv[0]);
	}
	switch (coder) {
	case uuencode:
	case b64encode:
		main_encode(argc, argv);
		break;
	case uudecode:
	case b64decode:
		main_decode(argc, argv);
		break;
	case base64:
		base64_encode_or_decode(argc, argv);
		break;
	case qp:
		main_quotedprintable(argc, argv);
		break;
	default:
		(void)fprintf(stderr,
		    "usage: %1$s <uuencode | uudecode> ...\n"
		    "       %1$s <b64encode | b64decode> ...\n"
		    "       %1$s <base64> ...\n"
		    "       %1$s <qp> ...\n",
		    progname);
		exit(EX_USAGE);
	}
}

static int
search(const char *const progname)
{
#define DESIGNATE(item) [item] = #item
	const char *const known[] = {
		DESIGNATE(uuencode),
		DESIGNATE(uudecode),
		DESIGNATE(b64encode),
		DESIGNATE(b64decode),
		DESIGNATE(base64),
		DESIGNATE(qp)
	};

	for (size_t i = 0; i < nitems(known); i++)
		if (strcmp(progname, known[i]) == 0)
			return ((int)i);
	return (-1);
}

static void
usage_base64(bool failure)
{
#ifdef __APPLE__
	(void)fputs(
"Usage:	base64 [-Ddh] [-b num] [-i in_file] [-o out_file]\n"
"  -b, --break       break encoded output up into lines of length num\n"
"  -D, -d, --decode  decode input\n"
"  -h, --help        display this message\n"
"  -i, --input       input file (default: \"-\" for stdin)\n"
"  -o, --output      output file (default: \"-\" for stdout)\n",
	    failure ? stderr : stdout);
	exit(failure ? EX_USAGE : EX_OK);
#else
	(void)fputs("usage: base64 [-w col | --wrap=col] "
	    "[-d | --decode] [FILE]\n"
	    "       base64 --help\n"
	    "       base64 --version\n", stderr);
	exit(failure ? EXIT_FAILURE : EXIT_SUCCESS);
#endif
}

static void
version_base64(void)
{
	(void)fputs("FreeBSD base64\n", stderr);
	exit(EXIT_SUCCESS);
}

static void
base64_encode_or_decode(int argc, char *argv[])
{
	int ch;
	bool decode = false;
#ifdef __APPLE__
	const char *infile = "-", *outfile = "-";
	/* default to no line breaks for compatibility */
	const char *w = "0";
#else
	const char *w = NULL;
#endif
	enum { HELP, VERSION };
	static const struct option opts[] =
	{
		{"decode",	no_argument,		NULL, 'd'},
#ifdef __APPLE__
		{"break",	required_argument,	NULL, 'b'},
		{"breaks",	required_argument,	NULL, 'b'},
		{"input",	required_argument,	NULL, 'i'},
		{"output",	required_argument,	NULL, 'o'},
#else
		{"ignore-garbage",no_argument,		NULL, 'i'},
#endif
		{"wrap",	required_argument,	NULL, 'w'},
		{"help",	no_argument,		NULL, HELP},
		{"version",	no_argument,		NULL, VERSION},
		{NULL,		no_argument,		NULL, 0}
	};

#ifdef __APPLE__
	while ((ch = getopt_long(argc, argv, "b:Ddhi:o:w:", opts, NULL)) != -1)
#else
	while ((ch = getopt_long(argc, argv, "diw:", opts, NULL)) != -1)
#endif
		switch (ch) {
#ifdef __APPLE__
		case 'D':
#endif
		case 'd':
			decode = true;
			break;
#ifdef __APPLE__
		case 'b':
#endif
		case 'w':
			w = optarg;
			break;
		case 'i':
#ifdef __APPLE__
			infile = optarg;
#else
			/* silently ignore */
#endif
			break;
#ifdef __APPLE__
		case 'o':
			outfile = optarg;
			break;
#endif
		case VERSION:
			version_base64();
			break;
#ifdef __APPLE__
		case 'h':
#endif
		case HELP:
		default:
			usage_base64(ch == '?');
		}

#ifdef __APPLE__
	argc -= optind;
	argv += optind;
	if (argc > 0) {
		warnx("invalid argument %s", *argv);
		usage_base64(true);
	}
	if (decode)
		main_base64_decode(infile, outfile);
	else
		main_base64_encode(infile, outfile, w);
#else
	if (decode)
		main_base64_decode(argv[optind]);
	else
		main_base64_encode(argv[optind], w);
#endif
}
