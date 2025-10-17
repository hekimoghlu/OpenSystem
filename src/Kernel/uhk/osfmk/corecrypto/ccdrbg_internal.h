/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#ifndef _CORECRYPTO_CCDRBG_INTERNAL_H_
#define _CORECRYPTO_CCDRBG_INTERNAL_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccdrbg_impl.h>
#include <corecrypto/ccdrbg.h>
#include <corecrypto/ccaes.h>

#define DRBG_CTR_KEYLEN(drbg)   ((drbg)->custom.keylen)
#define DRBG_CTR_CTRLEN         (8)
#define DRBG_CTR_BLOCKLEN(drbg) (CCAES_BLOCK_SIZE)
#define DRBG_CTR_SEEDLEN(drbg)  (DRBG_CTR_KEYLEN(drbg) + DRBG_CTR_BLOCKLEN(drbg))

#define DRBG_CTR_MAX_KEYLEN     (CCAES_KEY_SIZE_256)
#define DRBG_CTR_MAX_BLOCKLEN   (CCAES_BLOCK_SIZE)
#define DRBG_CTR_MAX_SEEDLEN    (DRBG_CTR_MAX_KEYLEN + DRBG_CTR_MAX_BLOCKLEN)

struct ccdrbg_nistctr_state {
	uint8_t Key[DRBG_CTR_MAX_KEYLEN];
	uint8_t V[DRBG_CTR_MAX_BLOCKLEN];
	uint64_t reseed_counter; // Fits max NIST requirement of 2^48.
	struct ccdrbg_nistctr_custom custom;
};

/*
 * NIST SP 800-90 TRNG DRBG
 *
 * Call into the SEP DRBG and perform a SP 800-90 test operation.
 */
void ccdrbg_factory_trng(struct ccdrbg_info *info);

/* Required length of the various TRNG entropy and personalization inputs. */
#define CCDRBG_TRNG_VECTOR_LEN     48

#endif /* _CORECRYPTO_CCDRBG_INTERNAL_H_ */
