/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

static int version_flag = 0;
static int help_flag	= 0;

static struct getargs args[] = {
    {"version",	0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void usage (int ret) HEIMDAL_NORETURN_ATTRIBUTE;

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args)/sizeof(*args),
		    NULL,
		    "[realms ...]");
    exit (ret);
}

int
main(int argc, char **argv)
{
    int i;
    size_t j;
    krb5_context context;
    int types[] = {KRB5_KRBHST_KDC, KRB5_KRBHST_ADMIN, KRB5_KRBHST_CHANGEPW,
		   KRB5_KRBHST_KRB524};
    const char *type_str[] = {"kdc", "admin", "changepw", "krb524"};
    int optidx = 0;

    setprogname (argv[0]);

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

    krb5_init_context (&context);
    for(i = 0; i < argc; i++) {
	krb5_krbhst_handle handle;
	char host[MAXHOSTNAMELEN];

	for (j = 0; j < sizeof(types)/sizeof(*types); ++j) {
	    printf ("%s for %s:\n", type_str[j], argv[i]);

	    krb5_krbhst_init(context, argv[i], types[j], &handle);
	    while(krb5_krbhst_next_as_string(context, handle,
					     host, sizeof(host)) == 0)
		printf("\thost: %s\n", host);
	    krb5_krbhst_reset(context, handle);
	    printf ("\n");
	}
    }
    return 0;
}
