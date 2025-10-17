/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#ifndef _CRYPTO_AESXTS_H
#define _CRYPTO_AESXTS_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include <corecrypto/ccmode.h>
#include <corecrypto/ccaes.h>
#include <corecrypto/ccn.h>

//Unholy HACK: this works because we know the size of the context for every
//possible corecrypto implementation is less than this.
#define AES_XTS_CTX_MAX_SIZE (ccn_sizeof_size(3*sizeof(void *)) + 2*ccn_sizeof_size(128*4) + ccn_sizeof_size(16))

typedef struct {
	ccxts_ctx_decl(AES_XTS_CTX_MAX_SIZE, enc);
	ccxts_ctx_decl(AES_XTS_CTX_MAX_SIZE, dec);
} symmetric_xts;


/*
 * These are the interfaces required for XTS-AES support
 */

uint32_t
xts_start(uint32_t cipher, // ignored - we're doing this for xts-aes only
    const uint8_t *IV,               // ignored
    const uint8_t *key1, int keylen,
    const uint8_t *key2, int tweaklen,               // both keys are the same size for xts
    uint32_t num_rounds,               // ignored
    uint32_t options,                  // ignored
    symmetric_xts *xts);

int xts_encrypt(const uint8_t *pt, unsigned long ptlen,
    uint8_t *ct,
    const uint8_t *tweak,                             // this can be considered the sector IV for this use
    symmetric_xts *xts);

int xts_decrypt(const uint8_t *ct, unsigned long ptlen,
    uint8_t *pt,
    const uint8_t *tweak,                             // this can be considered the sector IV for this use
    symmetric_xts *xts);

void xts_done(symmetric_xts *xts);

#if defined(__cplusplus)
}
#endif

#endif
