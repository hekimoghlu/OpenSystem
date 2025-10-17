/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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

static void
print_data(unsigned char *data, size_t len)
{
    int i;
    for(i = 0; i < len; i++) {
	if(i > 0 && (i % 16) == 0)
	    printf("\n            ");
	printf("%02x ", data[i]);
    }
    printf("\n");
}

static int
compare(const char *name, krb5_storage *sp, void *expected, size_t len)
{
    int ret = 0;
    krb5_data data;
    if (krb5_storage_to_data(sp, &data))
	errx(1, "krb5_storage_to_data failed");
    krb5_storage_free(sp);
    if(data.length != len || memcmp(data.data, expected, len) != 0) {
	printf("%s mismatch\n", name);
	printf("  Expected: ");
	print_data(expected, len);
	printf("  Actual:   ");
	print_data(data.data, data.length);
	ret++;
    }
    krb5_data_free(&data);
    return ret;
}

int
main(int argc, char **argv)
{
    int nerr = 0;
    krb5_storage *sp;
    krb5_context context;
    krb5_principal principal;


    krb5_init_context(&context);

    sp = krb5_storage_emem();
    krb5_store_int32(sp, 0x01020304);
    nerr += compare("Integer", sp, "\x1\x2\x3\x4", 4);

    sp = krb5_storage_emem();
    krb5_storage_set_byteorder(sp, KRB5_STORAGE_BYTEORDER_LE);
    krb5_store_int32(sp, 0x01020304);
    nerr += compare("Integer (LE)", sp, "\x4\x3\x2\x1", 4);

    sp = krb5_storage_emem();
    krb5_storage_set_byteorder(sp, KRB5_STORAGE_BYTEORDER_BE);
    krb5_store_int32(sp, 0x01020304);
    nerr += compare("Integer (BE)", sp, "\x1\x2\x3\x4", 4);

    sp = krb5_storage_emem();
    krb5_storage_set_byteorder(sp, KRB5_STORAGE_BYTEORDER_HOST);
    krb5_store_int32(sp, 0x01020304);
    {
	int test = 1;
	void *data;
	if(*(char*)&test)
	    data = "\x4\x3\x2\x1";
	else
	    data = "\x1\x2\x3\x4";
	nerr += compare("Integer (host)", sp, data, 4);
    }

    sp = krb5_storage_emem();
    krb5_make_principal(context, &principal, "TEST", "foobar", NULL);
    krb5_store_principal(sp, principal);
    krb5_free_principal(context, principal);
    nerr += compare("Principal", sp, "\x0\x0\x0\x1"
		    "\x0\x0\x0\x1"
		    "\x0\x0\x0\x4TEST"
		    "\x0\x0\x0\x6""foobar", 26);

    krb5_free_context(context);

    return nerr ? 1 : 0;
}
