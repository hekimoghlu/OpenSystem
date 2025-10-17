/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#ifndef _CORECRYPTO_CCCKG_H_
#define _CORECRYPTO_CCCKG_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccec.h>
#include <corecrypto/ccdigest.h>
#include <corecrypto/ccrng.h>

typedef struct __attribute__((aligned(16))) ccckg_ctx {
	ccec_const_cp_t cp;
}* ccckg_ctx_t;

size_t ccckg_sizeof_commitment(ccec_const_cp_t cp, const struct ccdigest_info* digest_info);

size_t ccckg_sizeof_share(ccec_const_cp_t cp, const struct ccdigest_info* digest_info);

size_t ccckg_sizeof_opening(ccec_const_cp_t cp, const struct ccdigest_info* digest_info);

size_t ccckg_sizeof_ctx(ccec_const_cp_t cp, const struct ccdigest_info* digest_info);

int ccckg_init(ccckg_ctx_t ctx, ccec_const_cp_t cp, const struct ccdigest_info* digest_info, struct ccrng_state* rng);

int ccckg_contributor_commit(ccckg_ctx_t ctx, size_t commitment_length, void* commitment);

int ccckg_contributor_finish(ccckg_ctx_t ctx, size_t share_length, const void* share, size_t opening_length, void* opening, ccec_pub_ctx_t ec_pub, size_t shared_key_length, void* shared_key);

int ccckg_owner_finish(ccckg_ctx_t ctx, size_t opening_length, const void* opening, ccec_full_ctx_t ec_full, size_t shared_key_length, void* shared_key);

int ccckg_owner_generate_share(ccckg_ctx_t ctx, size_t commitment_length, const void* commitment, size_t share_length, void* share);

#endif // _CORECRYPTO_CCCKG_H_
