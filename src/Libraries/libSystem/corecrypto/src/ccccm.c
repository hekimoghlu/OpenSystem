/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include <corecrypto/ccmode_factory.h>
#include <corecrypto/ccstubs.h>

int ccmode_ccm_init(const struct ccmode_ccm* ccm, ccccm_ctx* _ctx, size_t rawkey_len, const void* rawkey) {
	CC_STUB_ERR();
};

int ccmode_ccm_set_iv(ccccm_ctx* _ctx, ccccm_nonce* _nonce_ctx, size_t nonce_len, const void* nonce, size_t mac_size, size_t auth_len, size_t data_len) {
	CC_STUB_ERR();
};

int ccmode_ccm_cbcmac(ccccm_ctx* _ctx, ccccm_nonce* _nonce_ctx, size_t nbytes, const void* in) {
	CC_STUB_ERR();
};

int ccmode_ccm_decrypt(ccccm_ctx *_ctx, ccccm_nonce* _nonce_ctx, size_t nbytes, const void* in, void* out) {
	CC_STUB_ERR();
};

int ccmode_ccm_encrypt(ccccm_ctx *_ctx, ccccm_nonce* _nonce_ctx, size_t nbytes, const void* in, void* out) {
	CC_STUB_ERR();
};

int ccmode_ccm_finalize(ccccm_ctx* _ctx, ccccm_nonce* _nonce_ctx, void* mac) {
	CC_STUB_ERR();
};

int ccmode_ccm_reset(ccccm_ctx* _ctx, ccccm_nonce* _nonce_ctx) {
	CC_STUB_ERR();
};
