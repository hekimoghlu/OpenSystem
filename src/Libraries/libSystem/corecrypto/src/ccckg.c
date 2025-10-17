/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#include <corecrypto/ccckg.h>
#include <corecrypto/ccstubs.h>

size_t ccckg_sizeof_commitment(ccec_const_cp_t cp, const struct ccdigest_info* digest_info) {
	CC_STUB(0);
};

size_t ccckg_sizeof_share(ccec_const_cp_t cp, const struct ccdigest_info* digest_info) {
	CC_STUB(0);
};

size_t ccckg_sizeof_opening(ccec_const_cp_t cp, const struct ccdigest_info* digest_info) {
	CC_STUB(0);
};

size_t ccckg_sizeof_ctx(ccec_const_cp_t cp, const struct ccdigest_info* digest_info) {
	CC_STUB(0);
};

int ccckg_init(ccckg_ctx_t ctx, ccec_const_cp_t cp, const struct ccdigest_info* digest_info, struct ccrng_state* rng) {
	CC_STUB_ERR();
};

int ccckg_contributor_commit(ccckg_ctx_t ctx, size_t commitment_length, void* commitment) {
	CC_STUB_ERR();
};

int ccckg_contributor_finish(ccckg_ctx_t ctx, size_t share_length, const void* share, size_t opening_length, void* opening, ccec_pub_ctx_t ec_pub, size_t shared_key_length, void* shared_key) {
	CC_STUB_ERR();
};

int ccckg_owner_finish(ccckg_ctx_t ctx, size_t opening_length, const void* opening, ccec_full_ctx_t ec_full, size_t shared_key_length, void* shared_key) {
	CC_STUB_ERR();
};

int ccckg_owner_generate_share(ccckg_ctx_t ctx, size_t commitment_length, const void* commitment, size_t share_length, void* share) {
	CC_STUB_ERR();
};
