/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include <hex.h>
#include <err.h>

/*
 * key: string2key(aes256, "testkey", "testkey", default_params)
 * input: unhex(1122334455667788)
 * output: 58b594b8a61df6e9439b7baa991ff5c1
 *
 * key: string2key(aes128, "testkey", "testkey", default_params)
 * input: unhex(1122334455667788)
 * output: ffa2f823aa7f83a8ce3c5fb730587129
 */

int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    krb5_keyblock key;
    krb5_crypto crypto;
    size_t length;
    krb5_data input, output, output2;
    krb5_enctype etype = ETYPE_AES256_CTS_HMAC_SHA1_96;

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context %d", ret);

    ret = krb5_generate_random_keyblock(context, etype, &key);
    if (ret)
	krb5_err(context, 1, ret, "krb5_generate_random_keyblock");

    ret = krb5_crypto_prf_length(context, etype, &length);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_prf_length");

    ret = krb5_crypto_init(context, &key, 0, &crypto);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_init");

    input.data = rk_UNCONST("foo");
    input.length = 3;

    ret = krb5_crypto_prf(context, crypto, &input, &output);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_prf");

    ret = krb5_crypto_prf(context, crypto, &input, &output2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_crypto_prf");

    if (krb5_data_cmp(&output, &output2) != 0)
	krb5_errx(context, 1, "krb5_data_cmp");

    krb5_data_free(&output);
    krb5_data_free(&output2);

    krb5_crypto_destroy(context, crypto);

    krb5_free_keyblock_contents(context, &key);

    krb5_free_context(context);

    return 0;
}
