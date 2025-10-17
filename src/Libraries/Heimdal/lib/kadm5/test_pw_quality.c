/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include "kadm5_locl.h"
#include <getarg.h>

RCSID("$Id$");

static int version_flag;
static int help_flag;
static char *principal;
static char *password;

static struct getargs args[] = {
    { "principal", 0, arg_string, &principal },
    { "password", 0, arg_string, &password },
    { "version", 0, arg_flag, &version_flag },
    { "help", 0, arg_flag, &help_flag }
};
int num_args = sizeof(args) / sizeof(args[0]);

int
main(int argc, char **argv)
{
    krb5_error_code ret;
    krb5_context context;
    krb5_principal p;
    const char *s;
    krb5_data pw_data;

    krb5_program_setup(&context, argc, argv, args, num_args, NULL);

    if(help_flag)
	krb5_std_usage(0, args, num_args);
    if(version_flag) {
	print_version(NULL);
	exit(0);
    }

    if (principal == NULL)
	krb5_errx(context, 1, "no principal given");
    if (password == NULL)
	krb5_errx(context, 1, "no password given");

    ret = krb5_parse_name(context, principal, &p);
    if (ret)
	krb5_errx(context, 1, "krb5_parse_name: %s", principal);

    pw_data.data = password;
    pw_data.length = strlen(password);

    kadm5_setup_passwd_quality_check (context, NULL, NULL);
    ret = kadm5_add_passwd_quality_verifier(context, NULL);
    if (ret)
	krb5_errx(context, 1, "kadm5_add_passwd_quality_verifier");

    s = kadm5_check_password_quality (context, p, &pw_data);
    if (s)
	krb5_errx(context, 1, "kadm5_check_password_quality:\n%s", s);

    krb5_free_principal(context, p);
    krb5_free_context(context);

    return 0;
}
