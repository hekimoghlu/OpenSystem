/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

static void usage (int status) __attribute__((noreturn));

static const char *cache;
static const char *credential;
static const char *principal_str;
static int help_flag;
static int version_flag;
#ifndef NO_AFS
static int unlog_flag = 1;
#endif
static int dest_tkt_flag = 1;
static int all_flag = 0;

struct getargs args[] = {
    { "credential",	0,   arg_string, rk_UNCONST(&credential),
      "remove one credential", "principal" },
    { "cache",		'c', arg_string, rk_UNCONST(&cache), "cache to destroy", "cache" },
    { "principal",	'p', arg_string, &principal_str, "client credential to destroy", "principal" },
    { "all",		'A', arg_flag, &all_flag, "destroy all caches", NULL },
    { NULL,		'a', arg_flag, &all_flag, "destroy all caches" },
#ifndef NO_AFS
    { "unlog",		0,   arg_negative_flag, &unlog_flag,
      "do not destroy tokens", NULL },
#endif
    { "delete-v4",	0,   arg_negative_flag, &dest_tkt_flag,
      "do not destroy v4 tickets", NULL },
    { "version", 	0,   arg_flag, &version_flag, NULL, NULL },
    { "help",		'h', arg_flag, &help_flag, NULL, NULL}
};

int num_args = sizeof(args) / sizeof(args[0]);

static void
usage (int status)
{
    arg_printusage (args, num_args, NULL, "");
    exit (status);
}

int
main (int argc, char **argv)
{
    krb5_error_code ret;
    krb5_context context;
    krb5_ccache  ccache;
    int optidx = 0;
    int exit_val = 0;

    setprogname (argv[0]);

    if(getarg(args, num_args, argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    argc -= optidx;

    if (argc != 0)
	usage (1);

    ret = krb5_init_context (&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    if (all_flag) {
	krb5_cccol_cursor cursor;

	ret = krb5_cccol_cursor_new (context, &cursor);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_cccol_cursor_new");

	while (krb5_cccol_cursor_next (context, cursor, &ccache) == 0 && ccache != NULL) {

	    ret = krb5_cc_destroy (context, ccache);
	    if (ret) {
		krb5_warn(context, ret, "krb5_cc_destroy");
		exit_val = 1;
	    }
	}
	krb5_cccol_cursor_free(context, &cursor);

    } else {
	if (cache != NULL && principal_str != NULL)
	    krb5_errx(context, 1, "Can't select on credential "
		      "and principal at the same time");

	if (principal_str) {
	    krb5_principal p;

	    ret = krb5_parse_name(context, principal_str, &p);
	    if (ret)
		krb5_err(context, 1, ret, "can't parse %s", principal_str);
	    
	    ret = krb5_cc_cache_match(context, p, &ccache);
	    krb5_free_principal(context, p);
	    if (ret)
		krb5_err(context, 1, ret, "Can't find cache for %s",
			 principal_str);
	} else if(cache == NULL) {
	    ret = krb5_cc_default(context, &ccache);
	    if (ret)
		krb5_err(context, 1, ret, "krb5_cc_default");
	} else {
	    ret =  krb5_cc_resolve(context,
				   cache,
				   &ccache);
	    if (ret)
		krb5_err(context, 1, ret, "krb5_cc_resolve");
	}

	if (ret == 0) {
	    if (credential) {
		krb5_creds mcred;

		krb5_cc_clear_mcred(&mcred);

		ret = krb5_parse_name(context, credential, &mcred.server);
		if (ret)
		    krb5_err(context, 1, ret,
			     "Can't parse principal %s", credential);

		ret = krb5_cc_remove_cred(context, ccache, 0, &mcred);
		if (ret)
		    krb5_err(context, 1, ret,
			     "Failed to remove principal %s", credential);

		krb5_cc_close(context, ccache);
		krb5_free_principal(context, mcred.server);
		krb5_free_context(context);
		return 0;
	    }

	    ret = krb5_cc_destroy (context, ccache);
	    if (ret) {
		krb5_warn(context, ret, "krb5_cc_destroy");
		exit_val = 1;
	    }
	}
    }

    krb5_free_context (context);

#ifndef NO_AFS
    if (unlog_flag && k_hasafs ()) {
	if (k_unlog ())
	    exit_val = 1;
    }
#endif

    return exit_val;
}
