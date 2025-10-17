/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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

#include <corecrypto/ccdh.h>
#include <corecrypto/ccstubs.h>

int ccdh_compute_key(ccdh_full_ctx_t private_key, ccdh_pub_ctx_t public_key,
                     cc_unit *computed_key) {
	CC_STUB_ERR();
}

int ccdh_import_pub(ccdh_const_gp_t gp, size_t in_len, const uint8_t *in,
                    ccdh_pub_ctx_t key) {
	CC_STUB_ERR();
}

int ccdh_generate_key(ccdh_const_gp_t gp, struct ccrng_state *rng,
                      ccdh_full_ctx_t key) {
	CC_STUB_ERR();
}

cc_size ccder_decode_dhparam_n(const uint8_t *der, const uint8_t *der_end) {
	CC_STUB(0);
}

const uint8_t *ccder_decode_dhparams(ccdh_gp_t gp, const uint8_t *der, const uint8_t *der_end) {
	CC_STUB(NULL);
}

void ccdh_export_pub(ccdh_pub_ctx_t key, void *out) {
	CC_STUB_VOID();
}

int ccdh_init_gp(ccdh_gp_t gp, cc_size n, cc_unit *p, cc_unit *g, cc_size l) {
	CC_STUB_ERR();
}

int ccdh_compute_shared_secret(ccdh_full_ctx_t ctx, ccdh_pub_ctx_t pub, size_t* len, const void* key, struct ccrng_state* rng) {
	CC_STUB_ERR();
};
