/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

static char *mkey_file;
static int help_flag;
static int version_flag;

struct getargs args[] = {
    { "mkey-file",	0,      arg_string, &mkey_file },
    { "help",		'h',	arg_flag,   &help_flag },
    { "version",	0,	arg_flag,   &version_flag }
};

static int num_args = sizeof(args) / sizeof(args[0]);

int
main(int argc, char **argv)
{
    krb5_context context;
    int ret, o = 0;

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
	errx(1, "krb5_init_context failed: %d", ret);

    if (mkey_file) {
        hdb_master_key mkey;

	ret = hdb_read_master_key(context, mkey_file, &mkey);
	if (ret)
	    krb5_err(context, 1, ret, "failed to read master key %s", mkey_file);

	hdb_free_master_key(context, mkey);
    } else
      krb5_errx(context, 1, "no command option given");

    krb5_free_context(context);

    return 0;
}
