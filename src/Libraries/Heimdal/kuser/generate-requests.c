/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#include "kuser_locl.h"

static unsigned
read_words (const char *filename, char ***ret_w)
{
    unsigned n, alloc;
    FILE *f;
    char buf[256];
    char **w = NULL;

    f = fopen (filename, "r");
    if (f == NULL)
	err (1, "cannot open %s", filename);
    alloc = n = 0;
    while (fgets (buf, sizeof(buf), f) != NULL) {
	buf[strcspn(buf, "\r\n")] = '\0';
	if (n >= alloc) {
	    alloc += 16;
	    w = erealloc (w, alloc * sizeof(char **));
	}
	w[n++] = estrdup (buf);
    }
    *ret_w = w;
    if (n == 0)
	errx(1, "%s is an empty file, no words to try", filename);
    fclose(f);
    return n;
}

static void
generate_requests (const char *filename, unsigned nreq)
{
    krb5_principal client;
    krb5_context context;
    krb5_error_code ret;
    krb5_creds cred;
    int i;
    char **words;
    unsigned nwords;

    ret = krb5_init_context (&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    nwords = read_words (filename, &words);

    for (i = 0; i < nreq; ++i) {
	char *name = words[rand() % nwords];

	memset(&cred, 0, sizeof(cred));

	ret = krb5_parse_name (context, name, &client);
	if (ret)
	    krb5_err (context, 1, ret, "krb5_parse_name %s", name);

	ret = krb5_get_init_creds_password (context, &cred, client, "",
					    NULL, NULL, 0, NULL, NULL);
	if (ret)
	    krb5_free_cred_contents (context, &cred);
	krb5_free_principal(context, client);
    }
}

static int version_flag	= 0;
static int help_flag	= 0;

static struct getargs args[] = {
    { "version", 	0,   arg_flag, &version_flag },
    { "help",		0,   arg_flag, &help_flag }
};

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args)/sizeof(*args),
		    NULL,
		    "file number");
    exit (ret);
}

int
main(int argc, char **argv)
{
    int optidx = 0;
    int nreq;
    char *end;

    setprogname(argv[0]);
    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag) {
	print_version(NULL);
	exit(0);
    }

    argc -= optidx;
    argv += optidx;

    if (argc != 2)
	usage (1);
    srand (0);
    nreq = strtol (argv[1], &end, 0);
    if (argv[1] == end || *end != '\0')
	usage (1);
    generate_requests (argv[0], nreq);
    return 0;
}
