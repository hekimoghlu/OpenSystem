/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef _CORECRYPTO_CCSHA1_H_
#define _CORECRYPTO_CCSHA1_H_

#include <corecrypto/ccdigest.h>
#include <corecrypto/cc_config.h>

#define CCSHA1_BLOCK_SIZE   64
#define CCSHA1_OUTPUT_SIZE  20
#define CCSHA1_STATE_SIZE   20

/* sha1 selector */
const struct ccdigest_info *ccsha1_di(void);

extern const uint32_t ccsha1_initial_state[5];

/* shared between several implementations */
void ccsha1_final(const struct ccdigest_info *di, ccdigest_ctx_t,
                  unsigned char *digest);


/* Implementations */
extern const struct ccdigest_info ccsha1_ltc_di;
extern const struct ccdigest_info ccsha1_eay_di;

#if  CCSHA1_VNG_INTEL
//extern const struct ccdigest_info ccsha1_vng_intel_di;
#if defined(__x86_64__)
extern const struct ccdigest_info ccsha1_vng_intel_AVX2_di;
extern const struct ccdigest_info ccsha1_vng_intel_AVX1_di;
#endif
extern const struct ccdigest_info ccsha1_vng_intel_SupplementalSSE3_di;
#endif

#if  CCSHA1_VNG_ARMV7NEON
extern const struct ccdigest_info ccsha1_vng_armv7neon_di;
#endif

/* TODO: Placeholders */
#define ccoid_sha1 ((unsigned char *)"\x06\x05\x2b\x0e\x03\x02\x1a")
#define ccoid_sha1_len 7

#endif /* _CORECRYPTO_CCSHA1_H_ */
