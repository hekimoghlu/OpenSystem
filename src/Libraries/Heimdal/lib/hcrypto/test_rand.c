/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
#include <config.h>

#include <stdio.h>

#include <roken.h>
#include <getarg.h>

#include "rand.h"


/*
 *
 */

static int version_flag;
static int help_flag;
static int len = 1024 * 1024;
static char *rand_method;
static char *filename;

static struct getargs args[] = {
    { "length",	0,	arg_integer,	&len,
      "length", NULL },
    { "file",	0,	arg_string,	&filename,
      "file name", NULL },
    { "method",	0,	arg_string,	&rand_method,
      "method", NULL },
    { "version",	0,	arg_flag,	&version_flag,
      "print version", NULL },
    { "help",		0,	arg_flag,	&help_flag,
      NULL, 	NULL }
};

/*
 *
 */

/*
 *
 */

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args)/sizeof(args[0]),
		    NULL,
		    "");
    exit (ret);
}

int
main(int argc, char **argv)
{
    int idx = 0;
    char *buffer;
    char path[MAXPATHLEN];

    setprogname(argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &idx))
	usage(1);

    if (help_flag)
	usage(0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    argc -= idx;
    argv += idx;

    if (argc != 0)
	usage(1);

    buffer = emalloc(len);

    if (rand_method) {
	if (0) {
	}
#ifndef NO_RAND_FORTUNA_METHOD
	else if (strcasecmp(rand_method, "fortuna") == 0)
	    RAND_set_rand_method(RAND_fortuna_method());
#endif
#ifndef NO_RAND_UNIX_METHOD
	else if (strcasecmp(rand_method, "unix") == 0)
	    RAND_set_rand_method(RAND_unix_method());
#endif
#ifndef __APPLE_PRIVATE__
	else if (strcasecmp(rand_method, "cc") == 0)
	    RAND_set_rand_method(RAND_cc_method());
#endif
#ifndef NO_RAND_EGD_METHOD
	else if (strcasecmp(rand_method, "egd") == 0)
	    RAND_set_rand_method(RAND_egd_method());
#endif
#ifdef WIN32
	else if (strcasecmp(rand_method, "w32crypto") == 0)
	    RAND_set_rand_method(RAND_w32crypto_method());
#endif
	else
	    errx(1, "unknown method %s", rand_method);
    }

    if (RAND_file_name(path, sizeof(path)) == NULL)
	errx(1, "RAND_file_name failed");

    if (RAND_status() != 1)
	errx(1, "random not ready yet");

    if (RAND_bytes(buffer, len) != 1)
	errx(1, "RAND_bytes");

    if (filename)
	rk_dumpdata(filename, buffer, len);

    /* head vs tail */
    if (len >= 100000) {
	int bit, i;
	double res;
	int bits[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	for (i = 0; i < len; i++) {
	    unsigned char c = ((unsigned char *)buffer)[i];
	    for (bit = 0; bit < 8 && c; bit++) {
		if (c & 1)
		    bits[bit]++;
		c = c >> 1;
	    }
	}

	for (bit = 0; bit < 8; bit++) {

	    res = ((double)abs(len - bits[bit] * 2)) / (double)len;
	    if (res > 0.005)
		errx(1, "head%d vs tail%d > 0.5%%%% %lf == %d vs %d",
		     bit, bit, res, len, bits[bit]);

	    printf("head vs tails bit%d: %lf\n", bit, res);
	}
    }

    free(buffer);

    /* test write random file */
    {
	static const char *file = "test.file";
	if (RAND_write_file(file) != 1)
	    errx(1, "RAND_write_file");
	if (RAND_load_file(file, 1024) != 1)
	    errx(1, "RAND_load_file");
	unlink(file);
    }

    return 0;
}
