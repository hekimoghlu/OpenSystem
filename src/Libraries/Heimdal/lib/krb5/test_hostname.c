/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include "krb5_locl.h"
#include <err.h>
#include <getarg.h>

static int debug_flag	= 0;
static int version_flag = 0;
static int help_flag	= 0;

static int
expand_hostname(krb5_context context, const char *host)
{
    krb5_error_code ret;
    char *h, **r;

    ret = krb5_expand_hostname(context, host, &h);
    if (ret)
	krb5_err(context, 1, ret, "krb5_expand_hostname(%s)", host);

    free(h);

    if (debug_flag)
	printf("hostname: %s -> %s\n", host, h);

    ret = krb5_expand_hostname_realms(context, host, &h, &r);
    if (ret)
	krb5_err(context, 1, ret, "krb5_expand_hostname_realms(%s)", host);

    if (debug_flag) {
	int j;

	printf("hostname: %s -> %s\n", host, h);
	for (j = 0; r[j]; j++) {
	    printf("\trealm: %s\n", r[j]);
	}
    }
    free(h);
    krb5_free_host_realm(context, r);

    return 0;
}

static int
test_expand_hostname(krb5_context context)
{
    int i, errors = 0;

    struct t {
	krb5_error_code ret;
	const char *orig_hostname;
	const char *new_hostname;
    } tests[] = {
	{ 0, "pstn1.su.se", "pstn1.su.se" },
	{ 0, "pstnproxy.su.se", "pstnproxy.su.se" },
    };

    for (i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
	errors += expand_hostname(context, tests[i].orig_hostname);
    }

    return errors;
}

static struct getargs args[] = {
    {"debug",	'd',	arg_flag,	&debug_flag,
     "turn on debuggin", NULL },
    {"version",	0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args, sizeof(args)/sizeof(*args), NULL, "hostname ...");
    exit (ret);
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    int optidx = 0, errors = 0;

    setprogname(argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    argc -= optidx;
    argv += optidx;

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    if (argc > 0) {
	while (argc-- > 0)
	    errors += expand_hostname(context, *argv++);
	return errors;
    }

    errors += test_expand_hostname(context);

    krb5_free_context(context);

    return errors;
}
