/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

static void
usage (int ret)
{
    arg_printusage (args,
		    sizeof(args)/sizeof(*args),
		    NULL,
		    "[principal]");
    exit (ret);
}

int
main(int argc, char **argv)
{
    krb5_principal client;
    krb5_context context;
    const char *in_tkt_service = NULL;
    krb5_ccache id;
    krb5_error_code ret;
    krb5_creds out;
    int optidx = 0;

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

    if (argc > 0)
	in_tkt_service = argv[0];

    memset(&out, 0, sizeof(out));

    ret = krb5_init_context(&context);
    if (ret)
	krb5_err(context, 1, ret, "krb5_init_context");

    ret = krb5_cc_default(context, &id);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_default");

    ret = krb5_cc_get_principal(context, id, &client);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_default");

    ret = krb5_get_renewed_creds(context,
				 &out,
				 client,
				 id,
				 in_tkt_service);

    if(ret)
	krb5_err(context, 1, ret, "krb5_get_renewed_creds");

    if (krb5_principal_compare(context, out.client, client) != TRUE)
	krb5_errx(context, 1, "return principal is not as expected");

    krb5_free_cred_contents(context, &out);

    krb5_free_context(context);

    return 0;
}
