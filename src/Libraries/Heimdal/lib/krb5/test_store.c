/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

static void
test_int8(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    int8_t val[] = {
	0, 1, -1, 128, -127
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_int8(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_int8");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_int8(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_int8");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}

static void
test_int16(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    int16_t val[] = {
	0, 1, -1, 32768, -32767
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_int16(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_int16");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_int16(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_int16");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}

static void
test_int32(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    int32_t val[] = {
	0, 1, -1, 2147483647, -2147483646
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_int32(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_int32");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_int32(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_int32");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}

static void
test_uint8(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    uint8_t val[] = {
	0, 1, 255
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_uint8(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_uint8");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_uint8(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_uint8");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}

static void
test_uint16(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    uint16_t val[] = {
	0, 1, 65535
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_uint16(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_uint16");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_uint16(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_uint16");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}

static void
test_uint32(krb5_context context, krb5_storage *sp)
{
    krb5_error_code ret;
    int i;
    uint32_t val[] = {
	0, 1, 4294967295UL
    }, v;

    krb5_storage_truncate(sp, 0);

    for (i = 0; i < sizeof(val[0])/sizeof(val); i++) {

	ret = krb5_store_uint32(sp, val[i]);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_store_uint32");
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_uint32(sp, &v);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_ret_uint32");
	if (v != val[i])
	    krb5_errx(context, 1, "store and ret mismatch");
    }
}


static void
test_storage(krb5_context context, krb5_storage *sp)
{
    test_int8(context, sp);
    test_int16(context, sp);
    test_int32(context, sp);
    test_uint8(context, sp);
    test_uint16(context, sp);
    test_uint32(context, sp);
}


static void
test_truncate(krb5_context context, krb5_storage *sp, int fd)
{
    struct stat sb;

    krb5_store_string(sp, "hej");
    krb5_storage_truncate(sp, 2);

    if (fstat(fd, &sb) != 0)
	krb5_err(context, 1, errno, "fstat");
    if (sb.st_size != 2)
	krb5_errx(context, 1, "length not 2");

    krb5_storage_truncate(sp, 1024);

    if (fstat(fd, &sb) != 0)
	krb5_err(context, 1, errno, "fstat");
    if (sb.st_size != 1024)
	krb5_errx(context, 1, "length not 2");
}

static void
check_too_large(krb5_context context, krb5_storage *sp)
{
    uint32_t too_big_sizes[] = { INT_MAX, INT_MAX / 2, INT_MAX / 4, INT_MAX / 8 + 1};
    krb5_error_code ret;
    krb5_data data;
    size_t n;

    for (n = 0; n < sizeof(too_big_sizes) / sizeof(too_big_sizes); n++) {
	krb5_storage_truncate(sp, 0);
	krb5_store_uint32(sp, too_big_sizes[n]);
	krb5_storage_seek(sp, 0, SEEK_SET);
	ret = krb5_ret_data(sp, &data);
	if (ret != HEIM_ERR_TOO_BIG)
	    errx(1, "not too big: %lu", (unsigned long)n);
    }
}

/*
 *
 */

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
    int fd, optidx = 0;
    krb5_storage *sp;
    const char *fn = "test-store-data";

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

    ret = krb5_init_context (&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    /*
     * Test encoding/decoding of primotive types on diffrent backends
     */

    sp = krb5_storage_emem();
    if (sp == NULL)
	krb5_errx(context, 1, "krb5_storage_emem: no mem");

    test_storage(context, sp);
    check_too_large(context, sp);
    krb5_storage_free(sp);


    fd = open(fn, O_RDWR|O_CREAT|O_TRUNC, 0600);
    if (fd < 0)
	krb5_err(context, 1, errno, "open(%s)", fn);

    sp = krb5_storage_from_fd(fd);
    close(fd);
    if (sp == NULL)
	krb5_errx(context, 1, "krb5_storage_from_fd: %s no mem", fn);

    test_storage(context, sp);
    krb5_storage_free(sp);
    unlink(fn);

    /*
     * test truncate behavior
     */

    fd = open(fn, O_RDWR|O_CREAT|O_TRUNC, 0600);
    if (fd < 0)
	krb5_err(context, 1, errno, "open(%s)", fn);

    sp = krb5_storage_from_fd(fd);
    if (sp == NULL)
	krb5_errx(context, 1, "krb5_storage_from_fd: %s no mem", fn);

    test_truncate(context, sp, fd);
    krb5_storage_free(sp);
    close(fd);
    unlink(fn);

    krb5_free_context(context);

    return 0;
}
