/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
time_encryption(krb5_context context, size_t size,
		krb5_enctype etype, int iterations)
{
    struct timeval tv1, tv2;
    krb5_error_code ret;
    krb5_keyblock key;
    krb5_crypto crypto;
    krb5_data data;
    char *etype_name;
    void *buf;
    int i;

    ret = krb5_generate_random_keyblock(context, etype, &key);
    if (ret)
	krb5_err(context, 1, ret, "krb5_generate_random_keyblock");

    ret = krb5_enctype_to_string(context, etype, &etype_name);
    if (ret)
	krb5_err(context, 1, ret, "krb5_enctype_to_string");

    buf = malloc(size);
    if (buf == NULL)
	krb5_errx(context, 1, "out of memory");
    memset(buf, 0, size);

    ret = krb5_crypto_init(context, &key, 0, &crypto);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_init");

    gettimeofday(&tv1, NULL);

    for (i = 0; i < iterations; i++) {
	ret = krb5_encrypt(context, crypto, 0, buf, size, &data);
	if (ret)
	    krb5_err(context, 1, ret, "encrypt: %d", i);
	krb5_data_free(&data);
    }

    gettimeofday(&tv2, NULL);

    timevalsub(&tv2, &tv1);

    printf("%s size: %7lu iterations: %d time: %3ld.%06ld\n",
	   etype_name, (unsigned long)size, iterations,
	   (long)tv2.tv_sec, (long)tv2.tv_usec);

    free(buf);
    free(etype_name);
    krb5_crypto_destroy(context, crypto);
    krb5_free_keyblock_contents(context, &key);
}

static void
time_s2k(krb5_context context,
	 krb5_enctype etype,
	 const char *password,
	 krb5_salt salt,
	 int iterations)
{
    struct timeval tv1, tv2;
    krb5_error_code ret;
    krb5_keyblock key;
    krb5_data opaque;
    char *etype_name;
    int i;

    ret = krb5_enctype_to_string(context, etype, &etype_name);
    if (ret)
	krb5_err(context, 1, ret, "krb5_enctype_to_string");

    opaque.data = NULL;
    opaque.length = 0;

    gettimeofday(&tv1, NULL);

    for (i = 0; i < iterations; i++) {
	ret = krb5_string_to_key_salt_opaque(context, etype, password, salt,
					 opaque, &key);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_string_to_key_data_salt_opaque");
	krb5_free_keyblock_contents(context, &key);
    }

    gettimeofday(&tv2, NULL);

    timevalsub(&tv2, &tv1);

    printf("%s string2key %d iterations time: %3ld.%06ld\n",
	   etype_name, iterations, (long)tv2.tv_sec, (long)tv2.tv_usec);
    free(etype_name);

}

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
		    "");
    exit (ret);
}

int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    int enciter, s2kiter;
    size_t i;
    int optidx = 0;
    krb5_salt salt;

    krb5_enctype enctypes[] = {
	ETYPE_DES_CBC_CRC,
	ETYPE_DES3_CBC_SHA1,
	ETYPE_ARCFOUR_HMAC_MD5,
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

    salt.salttype = KRB5_PW_SALT;
    salt.saltvalue.data = NULL;
    salt.saltvalue.length = 0;

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    enciter = 1000;
    s2kiter = 100;

    for (i = 0; i < sizeof(enctypes)/sizeof(enctypes[0]); i++) {

	krb5_enctype_enable(context, enctypes[i]);

	time_encryption(context, 16, enctypes[i], enciter);
	time_encryption(context, 32, enctypes[i], enciter);
	time_encryption(context, 512, enctypes[i], enciter);
	time_encryption(context, 1024, enctypes[i], enciter);
	time_encryption(context, 2048, enctypes[i], enciter);
	time_encryption(context, 4096, enctypes[i], enciter);
	time_encryption(context, 8192, enctypes[i], enciter);
	time_encryption(context, 16384, enctypes[i], enciter);
	time_encryption(context, 32768, enctypes[i], enciter);

	time_s2k(context, enctypes[i], "mYsecreitPassword", salt, s2kiter);
    }

    krb5_free_context(context);

    return 0;
}
