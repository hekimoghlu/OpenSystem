/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "gsskrb5_locl.h"

struct range {
    size_t lower;
    size_t upper;
};

struct range tests[] = {
    { 0, 1040 },
    { 2040, 2080 },
    { 4080, 5000 },
    { 8180, 8292 },
    { 9980, 10010 }
};

static void
test_range(const struct range *r, int integ,
	   krb5_context context, krb5_crypto crypto)
{
    krb5_error_code ret;
    size_t size, rsize;
    struct gsskrb5_crypto ctx;

    for (size = r->lower; size < r->upper; size++) {
	size_t cksumsize;
	uint16_t padsize;
	OM_uint32 minor;
	OM_uint32 max_wrap_size;

	ctx.crypto = crypto;

	ret = _gssapi_wrap_size_cfx(&minor,
				    &ctx,
				    context,
				    integ,
				    0,
				    size,
				    &max_wrap_size);
	if (ret)
	    krb5_errx(context, 1, "_gsskrb5cfx_max_wrap_length_cfx: %d", ret);
	if (max_wrap_size == 0)
	    continue;

	ret = _gsskrb5cfx_wrap_length_cfx(context,
					  crypto,
					  integ,
					  max_wrap_size,
					  &rsize, &cksumsize, &padsize);
	if (ret)
	    krb5_errx(context, 1, "_gsskrb5cfx_wrap_length_cfx: %d", ret);

	if (size < rsize)
	    krb5_errx(context, 1,
		      "size (%d) < rsize (%d) for max_wrap_size %d",
		      (int)size, (int)rsize, (int)max_wrap_size);
    }
}

static void
test_special(krb5_context context, krb5_crypto crypto,
	     int integ, size_t testsize)
{
    krb5_error_code ret;
    size_t rsize;
    OM_uint32 max_wrap_size;
    size_t cksumsize;
    uint16_t padsize;
    struct gsskrb5_crypto ctx;
    OM_uint32 minor;

    ctx.crypto = crypto;

    ret = _gssapi_wrap_size_cfx(&minor,
				&ctx,
				context,
				integ,
				0,
				testsize,
				&max_wrap_size);
    if (ret)
      krb5_errx(context, 1, "_gsskrb5cfx_max_wrap_length_cfx: %d", ret);
    if (ret)
	krb5_errx(context, 1, "_gsskrb5cfx_max_wrap_length_cfx: %d", ret);

    ret = _gsskrb5cfx_wrap_length_cfx(context,
				      crypto,
				      integ,
				      max_wrap_size,
				      &rsize, &cksumsize, &padsize);
    if (ret)
	krb5_errx(context, 1, "_gsskrb5cfx_wrap_length_cfx: %d", ret);

    if (testsize < rsize)
	krb5_errx(context, 1,
		  "testsize (%d) < rsize (%d) for max_wrap_size %d",
		  (int)testsize, (int)rsize, (int)max_wrap_size);
}




int
main(int argc, char **argv)
{
    krb5_keyblock keyblock;
    krb5_error_code ret;
    krb5_context context;
    krb5_crypto crypto;
    int i;

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_context_init: %d", ret);

    ret = krb5_generate_random_keyblock(context,
					KRB5_ENCTYPE_AES256_CTS_HMAC_SHA1_96,
					&keyblock);
    if (ret)
	krb5_err(context, 1, ret, "krb5_generate_random_keyblock");

    ret = krb5_crypto_init(context, &keyblock, 0, &crypto);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_init");

    test_special(context, crypto, 1, 60);
    test_special(context, crypto, 0, 60);

    for (i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
	test_range(&tests[i], 1, context, crypto);
	test_range(&tests[i], 0, context, crypto);
    }

    krb5_free_keyblock_contents(context, &keyblock);
    krb5_crypto_destroy(context, crypto);
    krb5_free_context(context);

    return 0;
}
