/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
		    "hostname");
    exit (ret);
}

int
main(int argc, char **argv)
{
    const char *hostname;
    krb5_context context;
    krb5_auth_context ac;
    krb5_error_code ret;
    krb5_creds cred;
    krb5_ccache id;
    krb5_data data;
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

    if (argc < 1)
	usage(1);

    hostname = argv[0];

    memset(&cred, 0, sizeof(cred));

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = krb5_cc_default(context, &id);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_default failed");

    ret = krb5_auth_con_init(context, &ac);
    if (ret)
	krb5_err(context, 1, ret, "krb5_auth_con_init failed");

    krb5_auth_con_addflags(context, ac,
			   KRB5_AUTH_CONTEXT_CLEAR_FORWARDED_CRED, NULL);

    ret = krb5_cc_get_principal(context, id, &cred.client);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_get_principal");

    ret = krb5_make_principal(context,
			      &cred.server,
			      krb5_principal_get_realm(context, cred.client),
			      KRB5_TGS_NAME,
			      krb5_principal_get_realm(context, cred.client),
			      NULL);
    if (ret)
	krb5_err(context, 1, ret, "krb5_make_principal(server)");

    ret = krb5_get_forwarded_creds (context,
				    ac,
				    id,
				    KDC_OPT_FORWARDABLE,
				    hostname,
				    &cred,
				    &data);
    if (ret)
	krb5_err (context, 1, ret, "krb5_get_forwarded_creds");

    krb5_data_free(&data);
    krb5_free_context(context);

    return 0;
}
