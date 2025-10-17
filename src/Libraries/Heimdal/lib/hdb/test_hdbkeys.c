/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "hdb_locl.h"
#include <getarg.h>
#include <base64.h>

static int help_flag;
static int version_flag;
static int kvno_integer = 1;

struct getargs args[] = {
    { "kvno",		'd',	arg_integer, &kvno_integer },
    { "help",		'h',	arg_flag,   &help_flag },
    { "version",	0,	arg_flag,   &version_flag }
};

static int num_args = sizeof(args) / sizeof(args[0]);

int
main(int argc, char **argv)
{
    krb5_principal principal;
    krb5_context context;
    char *principal_str, *password_str, *str;
    int ret, o = 0;
    hdb_keyset keyset;
    size_t length, len;
    void *data;

    setprogname(argv[0]);

    if(getarg(args, num_args, argc, argv, &o))
	krb5_std_usage(1, args, num_args);

    if(help_flag)
	krb5_std_usage(0, args, num_args);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    if (argc != 3)
	errx(1, "username and password missing");

    principal_str = argv[1];
    password_str = argv[2];

    ret = krb5_parse_name (context, principal_str, &principal);
    if (ret)
	krb5_err (context, 1, ret, "krb5_parse_name %s", principal_str);

    memset(&keyset, 0, sizeof(keyset));

    keyset.kvno = kvno_integer;
    keyset.set_time = malloc(sizeof (*keyset.set_time));
    if (keyset.set_time == NULL)
	errx(1, "couldn't allocate set_time field of keyset");
    *keyset.set_time = time(NULL);

    ret = hdb_generate_key_set_password(context, principal, password_str, 0, NULL,
					NULL, &keyset.keys.val, &len);
    if (ret)
	krb5_err(context, 1, ret, "hdb_generate_key_set_password");
    keyset.keys.len = len;

    if (keyset.keys.len == 0)
	krb5_errx (context, 1, "hdb_generate_key_set_password length 0");

    krb5_free_principal (context, principal);

    ASN1_MALLOC_ENCODE(hdb_keyset, data, length, &keyset, &len, ret);
    if (ret)
	krb5_errx(context, 1, "encode keyset");
    if (len != length)
	krb5_abortx(context, "foo");

    krb5_free_context(context);

    ret = base64_encode(data, length, &str);
    if (ret < 0)
	errx(1, "base64_encode");

    printf("keyset: %s\n", str);

    free(data);

    return 0;
}
