/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include "cc_internal.h"
#include "cc_macros.h"
#include "fipspost_trace.h"
#include "ccmode_gcm_internal.h"
#include <corecrypto/ccmode.h>

size_t
ccgcm_context_size(const struct ccmode_gcm *mode)
{
	CC_ENSURE_DIT_ENABLED

	return mode->size;
}

size_t
ccgcm_block_size(const struct ccmode_gcm *mode)
{
	CC_ENSURE_DIT_ENABLED

	return mode->block_size;
}

int
ccgcm_init(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t key_nbytes,
    const void *cc_sized_by(key_nbytes)key)
{
	CC_ENSURE_DIT_ENABLED

	return mode->init(mode, ctx, key_nbytes, key);
}

int
ccgcm_init_with_iv(const struct ccmode_gcm *mode, ccgcm_ctx *ctx,
    size_t key_nbytes, const void *key,
    const void *iv)
{
	CC_ENSURE_DIT_ENABLED

	int rc;

	rc = ccgcm_init(mode, ctx, key_nbytes, key);
	if (rc == 0) {
		rc = ccgcm_set_iv(mode, ctx, CCGCM_IV_NBYTES, iv);
	}
	if (rc == 0) {
		_CCMODE_GCM_KEY(ctx)->flags |= CCGCM_FLAGS_INIT_WITH_IV;
	}
	return rc;
}

int
ccgcm_set_iv(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t iv_nbytes,
    const void *cc_sized_by(iv_nbytes)iv)
{
	CC_ENSURE_DIT_ENABLED

	return mode->set_iv(ctx, iv_nbytes, iv);
}

int
ccgcm_inc_iv(CC_UNUSED const struct ccmode_gcm *mode, ccgcm_ctx *ctx, void *iv)
{
	CC_ENSURE_DIT_ENABLED

	uint8_t *Y0 = CCMODE_GCM_KEY_Y_0(ctx);

	cc_require(_CCMODE_GCM_KEY(ctx)->state == CCMODE_GCM_STATE_IV, errOut);
	cc_require(_CCMODE_GCM_KEY(ctx)->flags & CCGCM_FLAGS_INIT_WITH_IV, errOut);

	inc_uint(Y0 + 4, 8);
	cc_memcpy(iv, Y0, CCGCM_IV_NBYTES);
	cc_memcpy(CCMODE_GCM_KEY_Y(ctx), Y0, CCGCM_BLOCK_NBYTES);
	ccmode_gcm_update_pad(ctx);

	_CCMODE_GCM_KEY(ctx)->state = CCMODE_GCM_STATE_AAD;

	return 0;

errOut:
	return CCMODE_INVALID_CALL_SEQUENCE;
}

int
ccgcm_aad(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t nbytes,
    const void *cc_sized_by(nbytes)additional_data)
{
	CC_ENSURE_DIT_ENABLED

	return mode->gmac(ctx, nbytes, additional_data);
}

int
ccgcm_gmac(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t nbytes,
    const void *cc_sized_by(nbytes)in)
{
	CC_ENSURE_DIT_ENABLED

	return mode->gmac(ctx, nbytes, in);
}

int
ccgcm_update(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t nbytes,
    const void *cc_sized_by(nbytes)in,
    void *cc_sized_by(nbytes)out)
{
	CC_ENSURE_DIT_ENABLED

	return mode->gcm(ctx, nbytes, in, out);
}

int
ccgcm_finalize(const struct ccmode_gcm *mode,
    ccgcm_ctx *ctx,
    size_t tag_nbytes,
    void *cc_sized_by(tag_nbytes)tag)
{
	CC_ENSURE_DIT_ENABLED

	return mode->finalize(ctx, tag_nbytes, tag);
}

int
ccgcm_reset(const struct ccmode_gcm *mode, ccgcm_ctx *ctx)
{
	CC_ENSURE_DIT_ENABLED

	return mode->reset(ctx);
}

int
ccgcm_one_shot(const struct ccmode_gcm *mode,
    size_t key_nbytes, const void *key,
    size_t iv_nbytes, const void *iv,
    size_t adata_nbytes, const void *adata,
    size_t nbytes, const void *in, void *out,
    size_t tag_nbytes, void *tag)
{
	CC_ENSURE_DIT_ENABLED

	    FIPSPOST_TRACE_EVENT;

	int rc = 0;

	ccgcm_ctx_decl(mode->size, ctx);
	rc = ccgcm_init(mode, ctx, key_nbytes, key); cc_require(rc == 0, errOut);
	rc = ccgcm_set_iv(mode, ctx, iv_nbytes, iv); cc_require(rc == 0, errOut);
	rc = ccgcm_aad(mode, ctx, adata_nbytes, adata); cc_require(rc == 0, errOut);
	rc = ccgcm_update(mode, ctx, nbytes, in, out); cc_require(rc == 0, errOut);
	rc = ccgcm_finalize(mode, ctx, tag_nbytes, tag); cc_require(rc == 0, errOut);

errOut:
	ccgcm_ctx_clear(mode->size, ctx);
	return rc;
}


//ccgcm_one_shot_legacy() is created because in the previous implementation of aes-gcm
//set_iv() could be skipped.
//In the new version of aes-gcm set_iv() cannot be skipped and IV length cannot
//be zero, as specified in FIPS.
//do not call ccgcm_one_shot_legacy() in any new application
int
ccgcm_set_iv_legacy(const struct ccmode_gcm *mode, ccgcm_ctx *key, size_t iv_nbytes, const void *iv)
{
	CC_ENSURE_DIT_ENABLED

	int rc = -1;

	if (iv_nbytes == 0 || iv == NULL) {
		/* must be in IV state */
		cc_require(_CCMODE_GCM_KEY(key)->state == CCMODE_GCM_STATE_IV, errOut); /* CRYPT_INVALID_ARG */

		// this is the net effect of setting IV to the empty string
		cc_clear(CCGCM_BLOCK_NBYTES, CCMODE_GCM_KEY_Y(key));
		ccmode_gcm_update_pad(key);
		cc_clear(CCGCM_BLOCK_NBYTES, CCMODE_GCM_KEY_Y_0(key));

		_CCMODE_GCM_KEY(key)->state = CCMODE_GCM_STATE_AAD;
		rc = 0;
	} else {
		rc = ccgcm_set_iv(mode, key, iv_nbytes, iv);
	}

errOut:
	return rc;
}

int
ccgcm_one_shot_legacy(const struct ccmode_gcm *mode,
    size_t key_nbytes, const void *key,
    size_t iv_nbytes, const void *iv,
    size_t adata_nbytes, const void *adata,
    size_t nbytes, const void *in, void *out,
    size_t tag_nbytes, void *tag)
{
	CC_ENSURE_DIT_ENABLED

	int rc = 0;

	ccgcm_ctx_decl(mode->size, ctx);
	rc = ccgcm_init(mode, ctx, key_nbytes, key); cc_require(rc == 0, errOut);
	rc = ccgcm_set_iv_legacy(mode, ctx, iv_nbytes, iv); cc_require(rc == 0, errOut);
	rc = ccgcm_aad(mode, ctx, adata_nbytes, adata); cc_require(rc == 0, errOut);
	rc = ccgcm_update(mode, ctx, nbytes, in, out); cc_require(rc == 0, errOut);
	rc = ccgcm_finalize(mode, ctx, tag_nbytes, tag); cc_require(rc == 0, errOut);

errOut:
	ccgcm_ctx_clear(mode->size, ctx);
	return rc;
}

void
inc_uint(uint8_t *buf, size_t nbytes)
{
	for (size_t i = 1; i <= nbytes; i += 1) {
		size_t j = nbytes - i;
		buf[j] = (uint8_t)(buf[j] + 1);
		if (buf[j] > 0) {
			return;
		}
	}
}

void
ccmode_gcm_update_pad(ccgcm_ctx *key)
{
	inc_uint(CCMODE_GCM_KEY_Y(key) + 12, 4);
	CCMODE_GCM_KEY_ECB(key)->ecb(CCMODE_GCM_KEY_ECB_KEY(key), 1,
	    CCMODE_GCM_KEY_Y(key),
	    CCMODE_GCM_KEY_PAD(key));
}

void
ccmode_gcm_aad_finalize(ccgcm_ctx *key)
{
	if (_CCMODE_GCM_KEY(key)->state == CCMODE_GCM_STATE_AAD) {
		if (_CCMODE_GCM_KEY(key)->aad_nbytes % CCGCM_BLOCK_NBYTES > 0) {
			ccmode_gcm_mult_h(key, CCMODE_GCM_KEY_X(key));
		}
		_CCMODE_GCM_KEY(key)->state = CCMODE_GCM_STATE_TEXT;
	}
}

