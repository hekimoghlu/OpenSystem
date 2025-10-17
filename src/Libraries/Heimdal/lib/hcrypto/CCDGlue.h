/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
/*
 * Compat glue with CommonCrypto Digest functions
 */
 
#ifndef	_HCRYPTO_COMMON_DIGEST_H
#define _HCRYPTO_COMMON_DIGEST_H

#ifdef __cplusplus
#define HCCC_CPP_BEGIN extern "C" {
#define HCCC_CPP_END }
#else
#define HCCC_CPP_BEGIN
#define HCCC_CPP_END
#endif

#include <krb5-types.h>

#define CCDigest hc_CCDigest
#define CCDigestCreate hc_CCDigestCreate
#define CCDigestUpdate hc_CCDigestUpdate
#define CCDigestFinal hc_CCDigestFinal
#define CCDigestDestroy hc_CCDigestDestroy
#define CCDigestReset hc_CCDigestReset
#define CCDigestBlockSize hc_CCDigestBlockSize
#define CCDigestOutputSize hc_CCDigestOutputSize

HCCC_CPP_BEGIN

typedef struct CCDigestCtx_s *CCDigestRef;
typedef const struct CCDigest_s *CCDigestAlg;


#define kCCDigestMD4 (&hc_kCCDigestMD4_s)
extern const struct CCDigest_s hc_kCCDigestMD4_s;
#define kCCDigestMD5 (&hc_kCCDigestMD5_s)
extern const struct CCDigest_s hc_kCCDigestMD5_s;
#define kCCDigestSHA1 (&hc_kCCDigestSHA1_s)
extern const struct CCDigest_s hc_kCCDigestSHA1_s;
#define kCCDigestSHA256 (&hc_kCCDigestSHA256_s)
extern const struct CCDigest_s hc_kCCDigestSHA256_s;
#define kCCDigestSHA384 (&hc_kCCDigestSHA384_s)
extern const struct CCDigest_s hc_kCCDigestSHA384_s;
#define kCCDigestSHA512 (&hc_kCCDigestSHA512_s)
extern const struct CCDigest_s hc_kCCDigestSHA512_s;


int		CCDigest(CCDigestAlg, const void *, size_t, void *);
CCDigestRef	CCDigestCreate(CCDigestAlg);
int		CCDigestUpdate(CCDigestRef, const void *, size_t);
int		CCDigestFinal(CCDigestRef, void *);
void		CCDigestDestroy(CCDigestRef);
void		CCDigestReset(CCDigestRef);
size_t		CCDigestBlockSize(CCDigestRef) ;
size_t		CCDigestOutputSize(CCDigestRef);
    

#define CC_MD4_DIGEST_LENGTH 16
#define CC_MD5_DIGEST_LENGTH 16
#define CC_SHA1_DIGEST_LENGTH 20
#define CC_SHA256_DIGEST_LENGTH 32
#define CC_SHA384_DIGEST_LENGTH 48
#define CC_SHA512_DIGEST_LENGTH 64

/*
 *
 */
#if 0
typedef struct CCRandom_s *CCRandomRef;

#define kCCRandomDefault ((CCRandomRef)NULL)

#define CCRandomCopyBytes hc_CCRandomCopyBytes

int
CCRandomCopyBytes(CCRandomRef, void *, size_t);
#endif
/*
 *
 */

#ifndef HAVE_CCDESISWEAKKEY

#define CCDesIsWeakKey hc_CCDesIsWeakKey
#define CCDesSetOddParity hc_CCDesSetOddParity
#define CCDesCBCCksum hc_CCDesCBCCksum

CCCryptorStatus
CCDesIsWeakKey(const void *key, size_t length);

void
CCDesSetOddParity(void *key, size_t Length);

uint32_t
CCDesCBCCksum(void *input, void *output,
	      size_t length, void *key, size_t keylen,
	      void *ivec);

#endif

HCCC_CPP_END

#endif	/* _HCRYPTO_COMMON_DIGEST_H */
