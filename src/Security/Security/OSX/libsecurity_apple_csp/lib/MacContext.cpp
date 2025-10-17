/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
 * MacContext.cpp - AppleCSPContext for HMACSHA1
 */

#include "MacContext.h"
#include <HMACSHA1.h>
#include <Security/cssmerr.h>
#include <CommonCrypto/CommonDigest.h>	/* for digest sizes */
#ifdef	CRYPTKIT_CSP_ENABLE
#include <security_cryptkit/HmacSha1Legacy.h>
#endif	/* CRYPTKIT_CSP_ENABLE */

MacContext::~MacContext()
{
	memset(&hmacCtx, 0, sizeof(hmacCtx));
}
	
/* called out from CSPFullPluginSession....
 * both generate and verify */
void MacContext::init(const Context &context, bool isSigning)
{
	CCHmacAlgorithm ccAlg;
	
	/* obtain key from context */
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	
	symmetricKeyBits(context, session(), mAlg, 
		isSigning ? CSSM_KEYUSE_SIGN : CSSM_KEYUSE_VERIFY,
		keyData, keyLen);
	uint32 minKey = 0;
	switch(mAlg) {
		case CSSM_ALGID_SHA512:
			minKey = HMAC_SHA_MIN_KEY_SIZE;
			mDigestSize = CC_SHA512_DIGEST_LENGTH;
			ccAlg = kCCHmacAlgSHA512;
			break;
		case CSSM_ALGID_SHA384:
			minKey = HMAC_SHA_MIN_KEY_SIZE;
			mDigestSize = CC_SHA384_DIGEST_LENGTH;
			ccAlg = kCCHmacAlgSHA384;
			break;
		case CSSM_ALGID_SHA256:
			minKey = HMAC_SHA_MIN_KEY_SIZE;
			mDigestSize = CC_SHA256_DIGEST_LENGTH;
			ccAlg = kCCHmacAlgSHA256;
			break;
		case CSSM_ALGID_SHA1HMAC:
			minKey = HMAC_SHA_MIN_KEY_SIZE;
			mDigestSize = CC_SHA1_DIGEST_LENGTH;
			ccAlg = kCCHmacAlgSHA1;
			break;
		case CSSM_ALGID_MD5HMAC:
			minKey = HMAC_MD5_MIN_KEY_SIZE;
			mDigestSize = CC_MD5_DIGEST_LENGTH;
			ccAlg = kCCHmacAlgMD5;
			break;
		default:
			assert(0);			// factory should not have called us
			CssmError::throwMe(CSSMERR_CSP_INVALID_ALGORITHM);
	}
	if((keyLen < minKey) || (keyLen > HMAC_MAX_KEY_SIZE)) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	CCHmacInit(&hmacCtx, ccAlg, keyData, keyLen);
}

void MacContext::update(const CssmData &data)
{
	CCHmacUpdate(&hmacCtx, data.data(), data.length());
}

/* generate only */
void MacContext::final(CssmData &out)
{
	if(out.length() < mDigestSize) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	CCHmacFinal(&hmacCtx, out.data());
	out.Length = mDigestSize;
}

/* verify only */
#define MAX_DIGEST_SIZE		CC_SHA512_DIGEST_LENGTH

void MacContext::final(const CssmData &in)
{
	unsigned char mac[MAX_DIGEST_SIZE];
	
	CCHmacFinal(&hmacCtx, mac);
	if(memcmp(mac, in.data(), mDigestSize)) {
		CssmError::throwMe(CSSMERR_CSP_VERIFY_FAILED);
	}
}

size_t MacContext::outputSize(bool final, size_t inSize)
{
	return mDigestSize;
}

#ifdef 	CRYPTKIT_CSP_ENABLE

MacLegacyContext::~MacLegacyContext()
{
	if(mHmac) {
		hmacLegacyFree(mHmac);
		mHmac = NULL;
	}
}
	
/* called out from CSPFullPluginSession....
 * both generate and verify: */
void MacLegacyContext::init(const Context &context, bool isSigning)
{
	if(mHmac == NULL) {
		mHmac = hmacLegacyAlloc();
		if(mHmac == NULL) {
			CssmError::throwMe(CSSMERR_CSP_MEMORY_ERROR);
		}
	}
	
	/* obtain key from context */
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	
	/* FIXME - this may require a different key alg */
	symmetricKeyBits(context, session(), CSSM_ALGID_SHA1HMAC, 
		isSigning ? CSSM_KEYUSE_SIGN : CSSM_KEYUSE_VERIFY,
		keyData, keyLen);
	if((keyLen < HMAC_SHA_MIN_KEY_SIZE) || (keyLen > HMAC_MAX_KEY_SIZE)) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	
	OSStatus ortn = hmacLegacyInit(mHmac, keyData, (UInt32)keyLen);
	if(ortn) {
		MacOSError::throwMe(ortn);
	}
}

void MacLegacyContext::update(const CssmData &data)
{
	OSStatus ortn = hmacLegacyUpdate(mHmac,
		data.data(),
		(UInt32)data.length());
	if(ortn) {
		MacOSError::throwMe(ortn);
	}
}

/* generate only */
void MacLegacyContext::final(CssmData &out)
{
	if(out.length() < kHMACSHA1DigestSize) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	hmacLegacyFinal(mHmac, out.data());
}

/* verify only */
void MacLegacyContext::final(const CssmData &in)
{
	unsigned char mac[kHMACSHA1DigestSize];
	hmacLegacyFinal(mHmac, mac);
	if(memcmp(mac, in.data(), kHMACSHA1DigestSize)) {
		CssmError::throwMe(CSSMERR_CSP_VERIFY_FAILED);
	}
}

size_t MacLegacyContext::outputSize(bool final, size_t inSize)
{
	return kHMACSHA1DigestSize;
}

#endif	/* CRYPTKIT_CSP_ENABLE */
