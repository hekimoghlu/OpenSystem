/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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

#ifdef __APPLE__
#include <Security/Security.h>
#endif

struct krb5_dh_moduli;
struct AlgorithmIdentifier;
struct _krb5_krb_auth_data;
struct _krb5_key_data;
struct _krb5_key_type;
struct _krb5_checksum_type;
struct _krb5_encryption_type;
struct _krb5_srv_query_ctx;
struct krb5_fast_state;
struct _krb5_srp_group;
struct _krb5_srp;

#include <heimbase.h>
#include <hx509.h>
#include <krb5-private.h>

static void usage (int ret) __attribute__((noreturn));


int version_flag	= 0;
int help_flag		= 0;


static struct getargs args[] = {
    { "version", 	0,   arg_flag, &version_flag },
    { "help",		0,   arg_flag, &help_flag }
};

static void
usage (int ret)
{
    arg_printusage_i18n (args,
			 sizeof(args)/sizeof(*args),
			 N_("Usage: ", ""),
			 NULL,
			 "[principal [command]]",
			 getarg_i18n);
    exit (ret);
}

int
main (int argc, char **argv)
{
    krb5_principal client = NULL, server = NULL;
    krb5_error_code ret;
    krb5_context context;
    int optidx = 0;
    krb5_ccache id;
    char password[512] = { 0 } ;

    setprogname (argv[0]);

    ret = krb5_init_context (&context);
    if (ret)
	errx(1, "krb5_init_context failed: %d", ret);

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

    if (argc != 1)
	krb5_errx(context, 1, "principal missing");

    ret = krb5_parse_name (context, argv[0], &client);
    if (ret)
	krb5_err (context, 1, ret, "krb5_parse_name(%s)", argv[0]);

    ret = krb5_cc_cache_match(context, client, &id);
    if (ret) {
#ifdef XCACHE_IS_API_CACHE
	ret = krb5_cc_new_unique(context, "XCACHE", NULL, &id);
#else
	ret = krb5_cc_new_unique(context, "KCM", NULL, &id);
#endif
	if (ret)
	    krb5_err (context, 1, ret, "krb5_cc_new_unique");
    }
    
    ret = krb5_cc_initialize(context, id, client);
    if (ret)
	krb5_err (context, 1, ret, "krb5_cc_initialize");
    

#if defined(__APPLE__) && !defined(__APPLE_TARGET_EMBEDDED__)
    {
	const char *realm;
	OSStatus osret;
	UInt32 length;
	void *buffer;
	char *name;

	realm = krb5_principal_get_realm(context, client);

	ret = krb5_unparse_name_flags(context, client,
				      KRB5_PRINCIPAL_UNPARSE_NO_REALM, &name);
	if (ret)
	    goto nopassword;

	osret = SecKeychainFindGenericPassword(NULL, (UInt32)strlen(realm), realm,
					       (UInt32)strlen(name), name,
					       &length, &buffer, NULL);
	free(name);
	if (osret != noErr)
	    goto nopassword;

	if (length < sizeof(password) - 1) {
	    memcpy(password, buffer, length);
	    password[length] = '\0';
	}
	SecKeychainItemFreeContent(NULL, buffer);

    nopassword:
	do { } while(0);
    }
#endif
    if (password[0] == 0) {
	char *p, *prompt;
	
	krb5_unparse_name (context, client, &p);
	asprintf (&prompt, "%s's Password: ", p);
	free (p);
	
	if (UI_UTIL_read_pw_string(password, sizeof(password)-1, prompt, 0)){
	    memset(password, 0, sizeof(password));
	    krb5_cc_destroy(context, id);
	    exit(1);
	}
	free (prompt);
    }
#ifdef HAVE_KCM
    ret = _krb5_kcm_get_initial_ticket(context, id, client, server, password);
#else
    ret = _krb5_xcc_get_initial_ticket(context, id, client, server, password);
#endif
    memset(password, 0, sizeof(password));
    if (ret) {
	krb5_cc_destroy(context, id);
	krb5_err (context, 1, ret, "_krb5_kcm_get_initial_ticket");
    }

    return 0;
}
