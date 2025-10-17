/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
#include <libkern/crypto/aesxts.h>
#include <corecrypto/ccmode.h>
#include <corecrypto/ccpad.h>
#include <kern/debug.h>

/*
 * These are the interfaces required for XTS-AES support
 */

uint32_t
xts_start(uint32_t cipher __unused, // ignored - we're doing this for xts-aes only
    const uint8_t *IV __unused,               // ignored
    const uint8_t *key1, int keylen,
    const uint8_t *key2, int tweaklen __unused,               // both keys are the same size for xts
    uint32_t num_rounds __unused,               // ignored
    uint32_t options __unused,                  // ignored
    symmetric_xts *xts)
{
	const struct ccmode_xts *enc, *dec;

	if (!g_crypto_funcs) {
		panic("%s: corecrypto not registered!", __FUNCTION__);
	}

	enc = g_crypto_funcs->ccaes_xts_encrypt;
	dec = g_crypto_funcs->ccaes_xts_decrypt;

	if (!enc || !dec) {
		panic("%s: xts mode not registered? enc=%p, dec=%p", __FUNCTION__, enc, dec);
	}

	/* Make sure the context size for the mode fits in the one we have */
	if ((enc->size > sizeof(xts->enc)) || (dec->size > sizeof(xts->dec))) {
		panic("%s: inconsistent size for AES-XTS context", __FUNCTION__);
	}

	int rc = enc->init(enc, xts->enc, keylen, key1, key2);
	rc |= dec->init(dec, xts->dec, keylen, key1, key2);

	return rc;
}

int
xts_encrypt(const uint8_t *pt, unsigned long ptlen,
    uint8_t *ct,
    const uint8_t *iv,                     // this can be considered the sector IV for this use
    symmetric_xts *xts)
{
	const struct ccmode_xts *xtsenc = g_crypto_funcs->ccaes_xts_encrypt;
	ccxts_tweak_decl(xtsenc->tweak_size, tweak);

	if (ptlen % 16) {
		panic("xts encrypt not a multiple of block size");
	}

	int rc = xtsenc->set_tweak(xts->enc, tweak, iv);
	if (rc) {
		return rc;
	}

	xtsenc->xts(xts->enc, tweak, ptlen / 16, pt, ct);
	return 0;
}

int
xts_decrypt(const uint8_t *ct, unsigned long ptlen,
    uint8_t *pt,
    const uint8_t *iv,                             // this can be considered the sector IV for this use
    symmetric_xts *xts)
{
	const struct ccmode_xts *xtsdec = g_crypto_funcs->ccaes_xts_decrypt;
	ccxts_tweak_decl(xtsdec->tweak_size, tweak);

	if (ptlen % 16) {
		panic("xts decrypt not a multiple of block size");
	}

	int rc = xtsdec->set_tweak(xts->dec, tweak, iv);
	if (rc) {
		return rc;
	}

	xtsdec->xts(xts->dec, tweak, ptlen / 16, ct, pt);
	return 0;
}

void
xts_done(symmetric_xts *xts __unused)
{
	cc_clear(sizeof(xts->enc), xts->enc);
	cc_clear(sizeof(xts->dec), xts->dec);
}
