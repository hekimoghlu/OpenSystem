/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include <libkern/crypto/crypto_internal.h>
#include <libkern/crypto/rsa.h>
#include <corecrypto/ccrsa.h>


int
rsa_make_pub(rsa_pub_ctx *pub,
    size_t exp_nbytes, const uint8_t *exp,
    size_t mod_nbytes, const uint8_t *mod)
{
	if ((exp_nbytes > RSA_MAX_KEY_BITSIZE / 8)
	    || (mod_nbytes > RSA_MAX_KEY_BITSIZE / 8)) {
		return -1; // Too big
	}
	ccrsa_ctx_n(pub->key) = ccn_nof(RSA_MAX_KEY_BITSIZE);
	return g_crypto_funcs->ccrsa_make_pub_fn(pub->key,
	           exp_nbytes, exp,
	           mod_nbytes, mod);
}

int
rsa_verify_pkcs1v15(rsa_pub_ctx *pub, const uint8_t *oid,
    size_t digest_len, const uint8_t *digest,
    size_t sig_len, const uint8_t *sig,
    bool *valid)
{
	return g_crypto_funcs->ccrsa_verify_pkcs1v15_fn(pub->key, oid,
	           digest_len, digest,
	           sig_len, sig, valid);
}
