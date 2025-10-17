/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#include <getarg.h>
#include <err.h>

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_std_usage(int code, struct getargs *args, int num_args)
{
    arg_printusage(args, num_args, NULL, "");
    exit(code);
}

KRB5_LIB_FUNCTION int KRB5_LIB_CALL
krb5_program_setup(krb5_context *context, int argc, char **argv,
		   struct getargs *args, int num_args,
		   void (KRB5_LIB_CALL *usage)(int, struct getargs*, int))
{
    krb5_error_code ret;
    int optidx = 0;

    if(usage == NULL)
	usage = krb5_std_usage;

    setprogname(argv[0]);
    ret = krb5_init_context(context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    if(getarg(args, num_args, argc, argv, &optidx))
	(*usage)(1, args, num_args);
    return optidx;
}
