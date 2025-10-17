/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include <corecrypto/ccaes.h>
#include <corecrypto/ccdes.h>
#include <corecrypto/ccpad.h>
#include <corecrypto/ccsha1.h>
#include <sys/malloc.h>

int corecrypto_available(void);

int
corecrypto_available(void)
{
	return g_crypto_funcs ? 1 : 0;
}

const struct ccmode_cbc  *
ccaes_cbc_decrypt_mode(void)
{
	if (g_crypto_funcs) {
		return g_crypto_funcs->ccaes_cbc_decrypt;
	}
	return NULL;
}

const struct ccmode_cbc  *
ccaes_cbc_encrypt_mode(void)
{
	if (g_crypto_funcs) {
		return g_crypto_funcs->ccaes_cbc_encrypt;
	}
	return NULL;
}

const struct ccmode_cbc  *
ccdes3_cbc_decrypt_mode(void)
{
	if (g_crypto_funcs) {
		return g_crypto_funcs->cctdes_cbc_decrypt;
	}
	return NULL;
}

const struct ccmode_cbc *
ccdes3_cbc_encrypt_mode(void)
{
	if (g_crypto_funcs) {
		return g_crypto_funcs->cctdes_cbc_encrypt;
	}
	return NULL;
}

size_t
ccpad_cts3_decrypt(const struct ccmode_cbc *cbc, cccbc_ctx *cbc_key,
    cccbc_iv *iv, size_t nbytes, const void *in, void *out)
{
	if (g_crypto_funcs) {
		return (*g_crypto_funcs->ccpad_cts3_decrypt_fn)(cbc, cbc_key, iv, nbytes, in, out);
	}
	return 0;
}

size_t
ccpad_cts3_encrypt(const struct ccmode_cbc *cbc, cccbc_ctx *cbc_key,
    cccbc_iv *iv, size_t nbytes, const void *in, void *out)
{
	if (g_crypto_funcs) {
		return (*g_crypto_funcs->ccpad_cts3_encrypt_fn)(cbc, cbc_key, iv, nbytes, in, out);
	}
	return 0;
}

const struct ccdigest_info *ccsha1_ltc_di_ptr;

const struct ccdigest_info *
ccsha1_di(void)
{
	if (g_crypto_funcs) {
		return g_crypto_funcs->ccsha1_di;
	}
	return NULL;
}

void
ccdes_key_set_odd_parity(void *key, unsigned long length)
{
	if (g_crypto_funcs) {
		(*g_crypto_funcs->ccdes_key_set_odd_parity_fn)(key, length);
	}
}
