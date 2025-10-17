/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#include <libkern/crypto/aes.h>
#include <corecrypto/ccmode.h>
#include <corecrypto/ccaes.h>
#include <kern/debug.h>

aes_rval
aes_encrypt_key(const unsigned char *key, int key_len, aes_encrypt_ctx cx[1])
{
	const struct ccmode_cbc *cbc = g_crypto_funcs->ccaes_cbc_encrypt;

	/* Make sure the context size for the mode fits in the one we have */
	if (cbc->size > sizeof(aes_encrypt_ctx)) {
		panic("%s: inconsistent size for AES encrypt context", __FUNCTION__);
	}

	return cccbc_init(cbc, cx[0].ctx, key_len, key);
}

aes_rval
aes_encrypt_cbc(const unsigned char *in_blk, const unsigned char *in_iv, unsigned int num_blk,
    unsigned char *out_blk, aes_encrypt_ctx cx[1])
{
	const struct ccmode_cbc *cbc = g_crypto_funcs->ccaes_cbc_encrypt;
	cccbc_iv_decl(cbc->block_size, ctx_iv);

	int rc = cccbc_set_iv(cbc, ctx_iv, in_iv);
	if (rc) {
		return rc;
	}

	return cccbc_update(cbc, cx[0].ctx, ctx_iv, num_blk, in_blk, out_blk); //Actually cbc encrypt.
}

#if defined (__i386__) || defined (__x86_64__) || defined (__arm64__)
/* This does one block of ECB, using the CBC implementation - this allow to use the same context for both CBC and ECB */
aes_rval
aes_encrypt(const unsigned char *in_blk, unsigned char *out_blk, aes_encrypt_ctx cx[1])
{
	return aes_encrypt_cbc(in_blk, NULL, 1, out_blk, cx);
}
#endif

aes_rval
aes_decrypt_key(const unsigned char *key, int key_len, aes_decrypt_ctx cx[1])
{
	const struct ccmode_cbc *cbc = g_crypto_funcs->ccaes_cbc_decrypt;

	/* Make sure the context size for the mode fits in the one we have */
	if (cbc->size > sizeof(aes_decrypt_ctx)) {
		panic("%s: inconsistent size for AES decrypt context", __FUNCTION__);
	}

	return cccbc_init(cbc, cx[0].ctx, key_len, key);
}

aes_rval
aes_decrypt_cbc(const unsigned char *in_blk, const unsigned char *in_iv, unsigned int num_blk,
    unsigned char *out_blk, aes_decrypt_ctx cx[1])
{
	const struct ccmode_cbc *cbc = g_crypto_funcs->ccaes_cbc_decrypt;
	cccbc_iv_decl(cbc->block_size, ctx_iv);

	int rc = cccbc_set_iv(cbc, ctx_iv, in_iv);
	if (rc) {
		return rc;
	}

	return cccbc_update(cbc, cx[0].ctx, ctx_iv, num_blk, in_blk, out_blk); //Actually cbc decrypt.
}

#if defined (__i386__) || defined (__x86_64__) || defined (__arm64__)
/* This does one block of ECB, using the CBC implementation - this allow to use the same context for both CBC and ECB */
aes_rval
aes_decrypt(const unsigned char *in_blk, unsigned char *out_blk, aes_decrypt_ctx cx[1])
{
	return aes_decrypt_cbc(in_blk, NULL, 1, out_blk, cx);
}
#endif

aes_rval
aes_encrypt_key128(const unsigned char *key, aes_encrypt_ctx cx[1])
{
	return aes_encrypt_key(key, 16, cx);
}

aes_rval
aes_decrypt_key128(const unsigned char *key, aes_decrypt_ctx cx[1])
{
	return aes_decrypt_key(key, 16, cx);
}


aes_rval
aes_encrypt_key256(const unsigned char *key, aes_encrypt_ctx cx[1])
{
	return aes_encrypt_key(key, 32, cx);
}

aes_rval
aes_decrypt_key256(const unsigned char *key, aes_decrypt_ctx cx[1])
{
	return aes_decrypt_key(key, 32, cx);
}

aes_rval
aes_encrypt_key_gcm(const unsigned char *key, int key_len, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_init(gcm, ctx, key_len, key);
}

aes_rval
aes_encrypt_key_with_iv_gcm(const unsigned char *key, int key_len, const unsigned char *in_iv, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return g_crypto_funcs->ccgcm_init_with_iv_fn(gcm, ctx, key_len, key, in_iv);
}

aes_rval
aes_encrypt_set_iv_gcm(const unsigned char *in_iv, unsigned int len, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_set_iv(gcm, ctx, len, in_iv);
}

aes_rval
aes_encrypt_reset_gcm(ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_reset(gcm, ctx);
}

aes_rval
aes_encrypt_inc_iv_gcm(unsigned char *out_iv, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return g_crypto_funcs->ccgcm_inc_iv_fn(gcm, ctx, out_iv);
}

aes_rval
aes_encrypt_aad_gcm(const unsigned char *aad, unsigned int aad_bytes, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_aad(gcm, ctx, aad_bytes, aad);
}

aes_rval
aes_encrypt_gcm(const unsigned char *in_blk, unsigned int num_bytes,
    unsigned char *out_blk, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_update(gcm, ctx, num_bytes, in_blk, out_blk);      //Actually gcm encrypt.
}

aes_rval
aes_encrypt_finalize_gcm(unsigned char *tag, size_t tag_bytes, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return aes_error;
	}

	int rc = ccgcm_finalize(gcm, ctx, tag_bytes, tag);
	if (rc) {
		return rc;
	}

	return ccgcm_reset(gcm, ctx);
}

aes_rval
aes_decrypt_key_gcm(const unsigned char *key, int key_len, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_init(gcm, ctx, key_len, key);
}

aes_rval
aes_decrypt_key_with_iv_gcm(const unsigned char *key, int key_len, const unsigned char *in_iv, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return g_crypto_funcs->ccgcm_init_with_iv_fn(gcm, ctx, key_len, key, in_iv);
}

aes_rval
aes_decrypt_set_iv_gcm(const unsigned char *in_iv, size_t len, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	int rc = ccgcm_reset(gcm, ctx);
	if (rc) {
		return rc;
	}

	return ccgcm_set_iv(gcm, ctx, len, in_iv);
}

aes_rval
aes_decrypt_reset_gcm(ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_reset(gcm, ctx);
}

aes_rval
aes_decrypt_inc_iv_gcm(unsigned char *out_iv, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return g_crypto_funcs->ccgcm_inc_iv_fn(gcm, ctx, out_iv);
}

aes_rval
aes_decrypt_aad_gcm(const unsigned char *aad, unsigned int aad_bytes, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_aad(gcm, ctx, aad_bytes, aad);
}

aes_rval
aes_decrypt_gcm(const unsigned char *in_blk, unsigned int num_bytes,
    unsigned char *out_blk, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	return ccgcm_update(gcm, ctx, num_bytes, in_blk, out_blk);      //Actually gcm decrypt.
}

aes_rval
aes_decrypt_finalize_gcm(unsigned char *tag, size_t tag_bytes, ccgcm_ctx *ctx)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return aes_error;
	}

	int rc = ccgcm_finalize(gcm, ctx, tag_bytes, tag);
	if (rc) {
		return rc;
	}

	return ccgcm_reset(gcm, ctx);
}

size_t
aes_encrypt_get_ctx_size_gcm(void)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_encrypt;
	if (!gcm) {
		return 0;
	}
	return cc_ctx_sizeof(ccgcm_ctx, gcm->size);
}

size_t
aes_decrypt_get_ctx_size_gcm(void)
{
	const struct ccmode_gcm *gcm = g_crypto_funcs->ccaes_gcm_decrypt;
	if (!gcm) {
		return 0;
	}
	return cc_ctx_sizeof(ccgcm_ctx, gcm->size);
}
