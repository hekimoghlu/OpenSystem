/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

static void
test_wrapping(krb5_context context,
	      size_t min_size,
	      size_t max_size,
	      size_t step,
	      krb5_enctype etype)
{
    krb5_error_code ret;
    krb5_keyblock key;
    krb5_crypto crypto;
    krb5_data data;
    char *etype_name;
    void *buf;
    size_t size;

    ret = krb5_generate_random_keyblock(context, etype, &key);
    if (ret)
	krb5_err(context, 1, ret, "krb5_generate_random_keyblock");

    ret = krb5_enctype_to_string(context, etype, &etype_name);
    if (ret)
	krb5_err(context, 1, ret, "krb5_enctype_to_string");

    buf = malloc(max_size);
    if (buf == NULL)
	krb5_errx(context, 1, "out of memory");
    memset(buf, 0, max_size);

    ret = krb5_crypto_init(context, &key, 0, &crypto);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_init");

    for (size = min_size; size < max_size; size += step) {
	size_t wrapped_size;

	ret = krb5_encrypt(context, crypto, 0, buf, size, &data);
	if (ret)
	    krb5_err(context, 1, ret, "encrypt size %lu using %s",
		     (unsigned long)size, etype_name);

	wrapped_size = krb5_get_wrapped_length(context, crypto, size);

	if (wrapped_size != data.length)
	    krb5_errx(context, 1, "calculated wrapped length %lu != "
		      "real wrapped length %lu for data length %lu using "
		      "enctype %s",
		      (unsigned long)wrapped_size,
		      (unsigned long)data.length,
		      (unsigned long)size,
		      etype_name);
	krb5_data_free(&data);
    }

    free(etype_name);
    free(buf);
    krb5_crypto_destroy(context, crypto);
    krb5_free_keyblock_contents(context, &key);
}



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
		    "");
    exit (ret);
}

int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    int i, optidx = 0;

    krb5_enctype enctypes[] = {
#ifdef HEIM_KRB5_DES
	ETYPE_DES_CBC_CRC,
	ETYPE_DES_CBC_MD4,
	ETYPE_DES_CBC_MD5,
#endif
#ifdef HEIM_KRB5_DES3
	ETYPE_DES3_CBC_SHA1,
#endif
#ifdef HEIM_KRB5_ARCFOUR
	ETYPE_ARCFOUR_HMAC_MD5,
#endif
	ETYPE_AES128_CTS_HMAC_SHA1_96,
	ETYPE_AES256_CTS_HMAC_SHA1_96
    };

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

    for (i = 0; i < sizeof(enctypes)/sizeof(enctypes[0]); i++) {
	krb5_enctype_enable(context, enctypes[i]);

	test_wrapping(context, 0, 1024, 1, enctypes[i]);
	test_wrapping(context, 1024, 1024 * 100, 1024, enctypes[i]);
    }
    krb5_free_context(context);

    return 0;
}
