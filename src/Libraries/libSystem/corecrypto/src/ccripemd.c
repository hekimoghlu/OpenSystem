/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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

#include <corecrypto/ccripemd.h>
#include <corecrypto/ccstubs.h>

#define USE_SUPER_COOL_NEW_CCOID_T
#include <corecrypto/ccripemd.h>
#include <corecrypto/ccmd4.h>
#include <corecrypto/cc_priv.h>
#include <corecrypto/ccdigest_priv.h>
#include <corecrypto/ccsha1.h>

static void rmd128_compress(ccdigest_state_t state, unsigned long nblocks, const void *in) {
	CC_STUB_VOID();
}

const struct ccdigest_info ccrmd128_ltc_di = {
    .output_size = CCRMD128_OUTPUT_SIZE,
    .state_size = CCRMD128_STATE_SIZE,
    .block_size = CCRMD_BLOCK_SIZE,
    .oid_size = 8,
    .oid = CC_DIGEST_OID_RMD128,
    .initial_state = ccmd4_initial_state,
    .compress = rmd128_compress,
    .final = ccdigest_final_64le,
};

static void rmd160_compress(ccdigest_state_t state, unsigned long nblocks, const void *in) {
	CC_STUB_VOID();
}

const struct ccdigest_info ccrmd160_ltc_di = {
    .output_size = CCRMD160_OUTPUT_SIZE,
    .state_size = CCRMD160_STATE_SIZE,
    .block_size = CCRMD_BLOCK_SIZE,
    .oid_size = 7,
    .oid = CC_DIGEST_OID_RMD160,
    .initial_state = ccsha1_initial_state,
    .compress = rmd160_compress,
    .final = ccdigest_final_64le,
};

static void rmd256_compress(ccdigest_state_t state, unsigned long nblocks, const void *in) {
	CC_STUB_VOID();
}

static const uint32_t ccrmd256_initial_state[8] = {
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
};

const struct ccdigest_info ccrmd256_ltc_di = {
    .output_size = CCRMD256_OUTPUT_SIZE,
    .state_size = CCRMD256_STATE_SIZE,
    .block_size = CCRMD_BLOCK_SIZE,
    .oid_size = 7,
    .oid = CC_DIGEST_OID_RMD256,
    .initial_state = ccrmd256_initial_state,
    .compress = rmd256_compress,
    .final = ccdigest_final_64le,
};

static void rmd320_compress(ccdigest_state_t state, unsigned long nblocks, const void *in) {
	CC_STUB_VOID();
}

static const uint32_t ccrmd320_initial_state[10] = {
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
};

const struct ccdigest_info ccrmd320_ltc_di = {
    .output_size = CCRMD320_OUTPUT_SIZE,
    .state_size = CCRMD320_STATE_SIZE,
    .block_size = CCRMD_BLOCK_SIZE,
    .oid_size = 0,
    .oid = CC_DIGEST_OID_RMD320,
    .initial_state = ccrmd320_initial_state,
    .compress = rmd320_compress,
    .final = ccdigest_final_64le,
};
