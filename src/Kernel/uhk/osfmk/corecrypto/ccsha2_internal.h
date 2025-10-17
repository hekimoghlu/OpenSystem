/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#ifndef _CORECRYPTO_CCSHA2_INTERNAL_H_
#define _CORECRYPTO_CCSHA2_INTERNAL_H_

#include <corecrypto/ccdigest.h>

#ifndef CCSHA2_DISABLE_SHA512
#define CCSHA2_DISABLE_SHA512 0
#endif

#define CCSHA2_SHA256_USE_SHA512_K (CC_SMALL_CODE && !CCSHA2_DISABLE_SHA512)

#if CCSHA256_ARMV6M_ASM
extern const struct ccdigest_info ccsha256_v6m_di;
void ccsha256_v6m_compress(ccdigest_state_t c, size_t num, const void *p) __asm__("_ccsha256_v6m_compress");
#endif

void ccsha256_ltc_compress(ccdigest_state_t state, size_t nblocks, const void *buf);
void ccsha512_ltc_compress(ccdigest_state_t state, size_t nblocks, const void *in);

#if CCSHA2_VNG_INTEL && defined(__x86_64__)
extern const struct ccdigest_info ccsha224_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha224_vng_intel_AVX1_di;
extern const struct ccdigest_info ccsha256_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha256_vng_intel_AVX1_di;
extern const struct ccdigest_info ccsha384_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha384_vng_intel_AVX1_di;
extern const struct ccdigest_info ccsha384_vng_intel_SupplementalSSE3_di;
extern const struct ccdigest_info ccsha512_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha512_vng_intel_AVX1_di;
extern const struct ccdigest_info ccsha512_vng_intel_SupplementalSSE3_di;
extern const struct ccdigest_info ccsha512_256_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha512_256_vng_intel_AVX1_di;
extern const struct ccdigest_info ccsha512_256_vng_intel_SupplementalSSE3_di;
#endif

#if CC_USE_L4
extern const struct ccdigest_info ccsha256_trng_di;
#endif

extern const uint32_t ccsha256_K[64];
extern const uint64_t ccsha512_K[80];

void ccsha512_final(const struct ccdigest_info *di, ccdigest_ctx_t ctx, unsigned char *digest);

extern const uint32_t ccsha224_initial_state[8];
extern const uint32_t ccsha256_initial_state[8];
extern const uint64_t ccsha384_initial_state[8];
extern const uint64_t ccsha512_initial_state[8];
extern const uint64_t ccsha512_256_initial_state[8];

#endif /* _CORECRYPTO_CCSHA2_INTERNAL_H_ */
