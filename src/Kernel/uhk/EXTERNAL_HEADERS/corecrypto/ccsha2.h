/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#ifndef _CORECRYPTO_CCSHA2_H_
#define _CORECRYPTO_CCSHA2_H_

#include <corecrypto/ccdigest.h>

CC_PTRCHECK_CAPABLE_HEADER()

/* sha2 selectors */
const struct ccdigest_info *ccsha224_di(void);
const struct ccdigest_info *ccsha256_di(void);
const struct ccdigest_info *ccsha384_di(void);
const struct ccdigest_info *ccsha512_di(void);
const struct ccdigest_info *ccsha512_256_di(void);  // SHA512/256 (cf FIPS 180-4 https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)

#define ccoid_sha224 ((unsigned char *)"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x04")
#define ccoid_sha224_len 11

#define ccoid_sha256 ((unsigned char *)"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x01")
#define ccoid_sha256_len 11

#define ccoid_sha384 ((unsigned char *)"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x02")
#define ccoid_sha384_len 11

#define ccoid_sha512 ((unsigned char *)"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x03")
#define ccoid_sha512_len 11

#define ccoid_sha512_256 ((unsigned char *)"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x06")
#define ccoid_sha512_256_len 11

/* SHA256 */
#define CCSHA256_BLOCK_SIZE  64
#define CCSHA256_OUTPUT_SIZE 32
#define CCSHA256_STATE_SIZE  32
extern const struct ccdigest_info ccsha256_ltc_di;
#if CCSHA2_VNG_INTEL
extern const struct ccdigest_info ccsha224_vng_intel_SupplementalSSE3_di;
extern const struct ccdigest_info ccsha256_vng_intel_SupplementalSSE3_di;
#endif
#if CCSHA2_VNG_ARM
extern const struct ccdigest_info ccsha224_vng_arm_di;
extern const struct ccdigest_info ccsha256_vng_arm_di;
#if defined(__arm64__)
extern const struct ccdigest_info ccsha256_vng_arm64neon_di;
#endif
extern const struct ccdigest_info ccsha384_vng_arm_di;
extern const struct ccdigest_info ccsha512_vng_arm_di;
extern const struct ccdigest_info ccsha512_256_vng_arm_di;
#endif

/* SHA224 */
#define	CCSHA224_OUTPUT_SIZE 28
extern const struct ccdigest_info ccsha224_ltc_di;

/* SHA512 */
#define CCSHA512_BLOCK_SIZE  128
#define	CCSHA512_OUTPUT_SIZE  64
#define	CCSHA512_STATE_SIZE   64
extern const struct ccdigest_info ccsha512_ltc_di;

/* SHA512/256 */
// FIPS 180-4
// https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
#define CCSHA512_256_BLOCK_SIZE  128
#define CCSHA512_256_OUTPUT_SIZE  32
#define CCSHA512_256_STATE_SIZE   64
extern const struct ccdigest_info ccsha512_256_ltc_di;

/* SHA384 */
#define	CCSHA384_OUTPUT_SIZE  48
extern const struct ccdigest_info ccsha384_ltc_di;

#endif /* _CORECRYPTO_CCSHA2_H_ */
