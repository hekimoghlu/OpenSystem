/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#include <libkern/libkern.h>
#include <kern/debug.h>
#include <libkern/crypto/des.h>
#include <corecrypto/ccmode.h>

/* Single DES ECB - used by ipv6 (esp_core.c) */
int
des_ecb_key_sched(des_cblock *key, des_ecb_key_schedule *ks)
{
	const struct ccmode_ecb *enc = g_crypto_funcs->ccdes_ecb_encrypt;
	const struct ccmode_ecb *dec = g_crypto_funcs->ccdes_ecb_decrypt;

	/* Make sure the context size for the mode fits in the one we have */
	if ((enc->size > sizeof(ks->enc)) || (dec->size > sizeof(ks->dec))) {
		panic("%s: inconsistent size for DES-ECB context", __FUNCTION__);
	}

	int rc = enc->init(enc, ks->enc, CCDES_KEY_SIZE, key);
	if (rc) {
		return rc;
	}

	return dec->init(dec, ks->dec, CCDES_KEY_SIZE, key);
}

/* Simple des - 1 block */
int
des_ecb_encrypt(des_cblock *in, des_cblock *out, des_ecb_key_schedule *ks, int enc)
{
	const struct ccmode_ecb *ecb = enc ? g_crypto_funcs->ccdes_ecb_encrypt : g_crypto_funcs->ccdes_ecb_decrypt;
	ccecb_ctx *ctx = enc ? ks->enc : ks->dec;

	return ecb->ecb(ctx, 1, in, out);
}


/* Triple DES ECB - used by ipv6 (esp_core.c) */
int
des3_ecb_key_sched(des_cblock *key, des3_ecb_key_schedule *ks)
{
	const struct ccmode_ecb *enc = g_crypto_funcs->cctdes_ecb_encrypt;
	const struct ccmode_ecb *dec = g_crypto_funcs->cctdes_ecb_decrypt;

	/* Make sure the context size for the mode fits in the one we have */
	if ((enc->size > sizeof(ks->enc)) || (dec->size > sizeof(ks->dec))) {
		panic("%s: inconsistent size for 3DES-ECB context", __FUNCTION__);
	}

	int rc = enc->init(enc, ks->enc, CCDES_KEY_SIZE * 3, key);
	if (rc) {
		return rc;
	}

	return dec->init(dec, ks->dec, CCDES_KEY_SIZE * 3, key);
}

/* Simple des - 1 block */
int
des3_ecb_encrypt(des_cblock *in, des_cblock *out, des3_ecb_key_schedule *ks, int enc)
{
	const struct ccmode_ecb *ecb = enc ? g_crypto_funcs->cctdes_ecb_encrypt : g_crypto_funcs->cctdes_ecb_decrypt;
	ccecb_ctx *ctx = enc ? ks->enc : ks->dec;

	return ecb->ecb(ctx, 1, in, out);
}

/* Raw key helper functions */

int
des_is_weak_key(des_cblock *key)
{
	return g_crypto_funcs->ccdes_key_is_weak_fn(key, CCDES_KEY_SIZE);
}
