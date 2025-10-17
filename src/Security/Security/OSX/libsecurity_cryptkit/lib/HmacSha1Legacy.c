/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
 	File:		HmacSha1Legacy.c
 	Contains:	HMAC/SHA1, bug-for-bug compatible with BSAFE 4.0.
 	Copyright (c) 2001,2011-2014 Apple Inc. All Rights Reserved.
*/

#include "ckconfig.h"

#include "HmacSha1Legacy.h"
#include "ckSHA1.h"
#include <string.h>
#include <stdlib.h>
#include <Security/SecBase.h>
#define kHMACSHA1DigestSize  20

/* XXX These should really be in ckSHA1.h */
#define kSHA1DigestSize  	20
#define kSHA1BlockSize  	64

/*
 * bug-for-bug compatible with BSAFE 4.0. See
 * BSafe/bsource/algs/ahchhmac.c.
 *
 * This implementation, and the BSAFE implementation it emulates, work fine 
 * when calculating a MAC in a single update (init, update, final). They 
 * generate nonconforming MACs when performing multiple updates because 
 * the entire algorithm - both inner and outer digests - are performed 
 * in the update() step. As a result, if one e.g. calculates a MAC of 
 * a block of text with one update, and then calculates the MAC over the 
 * same block of text via two updates, different results will obtain.ÊThe 
 * incorrect result from the multiple-update scenario is repeatable if and 
 * only if the same boundaries (same update sizes) are observed on each operation. 
 *
 * Because all of the data to be MAC'd is in fact protected by both levels of 
 * SHA1, and all of the key bits are used, this nonconforming implementation is
 * believed to be as strong, cryptographically, as a conforming SHA1HMAC
 * implementation. 
 */
struct hmacLegacyContext {
	sha1Obj sha1Context;
	UInt8 	k_ipad[kSHA1BlockSize];
	UInt8 	k_opad[kSHA1BlockSize];
};

hmacLegacyContextRef hmacLegacyAlloc(void)
{
	hmacLegacyContextRef hmac = 
		(hmacLegacyContextRef)malloc(sizeof(struct hmacLegacyContext));
	memset(hmac, 0, sizeof(struct hmacLegacyContext));
	return hmac;
}

void hmacLegacyFree(
	hmacLegacyContextRef hmac)
{
	if(hmac != NULL) {
		if(hmac->sha1Context != NULL) {
			sha1Free (hmac->sha1Context);
		}
		memset(hmac, 0, sizeof(struct hmacLegacyContext));
		free(hmac);
	}
}

/* reusable init */
OSStatus hmacLegacyInit(
	hmacLegacyContextRef hmac,
	const void *keyPtr,
	UInt32 keyLen)
{	
	UInt8 	*key;
	UInt32 	byte;

	if(hmac->sha1Context == NULL) {
		hmac->sha1Context = sha1Alloc();
		if(hmac->sha1Context == NULL) {
			return errSecAllocate;
		}
	}
	else {
		sha1Reinit(hmac->sha1Context);
	}
	/* this implementation requires a 20-byte key */
	if (keyLen != kSHA1DigestSize) {
		/* FIXME */
		return errSecParam;
	}
	key = (UInt8*)keyPtr;
	
	/* The HMAC_SHA_1 transform looks like:
	   SHA1 (K XOR opad || SHA1 (K XOR ipad || text))
	   Where K is a n byte key
	   ipad is the byte 0x36 repeated 64 times.
	   opad is the byte 0x5c repeated 64 times.
	   text is the data being protected.
	  */
	/* Copy the key into k_ipad and k_opad while doing the XOR. */
	for (byte = 0; byte < keyLen; byte++)
	{
		hmac->k_ipad[byte] = key[byte] ^ 0x36;
		hmac->k_opad[byte] = key[byte] ^ 0x5c;
	}

	/* Fill the remainder of k_ipad and k_opad with 0 XORed with 
	 * appropriate value. */
	memset (hmac->k_ipad + keyLen, 0x36, kSHA1BlockSize - keyLen);
	memset (hmac->k_opad + keyLen, 0x5c, kSHA1BlockSize - keyLen);
	
	/* remainder happens in update */
	return errSecSuccess;
}

OSStatus hmacLegacyUpdate(
	hmacLegacyContextRef hmac,
	const void *textPtr,
	UInt32 textLen)
{
	UInt8 innerDigest[kSHA1DigestSize];
	
	/* compute SHA1(k_ipad || data) ==> innerDigest */
	sha1AddData (hmac->sha1Context, hmac->k_ipad, kSHA1BlockSize);
	sha1AddData (hmac->sha1Context, (UInt8*)textPtr, textLen);
	memcpy (innerDigest, sha1Digest(hmac->sha1Context), kSHA1DigestSize);
	
	/* reset context (BSAFE does this implicitly in a final() call) */
	sha1Reinit(hmac->sha1Context);
	
	/* compute SHA1(k_opad || innerDigest) */
	sha1AddData (hmac->sha1Context, hmac->k_opad, kSHA1BlockSize);
	sha1AddData (hmac->sha1Context, innerDigest, kSHA1DigestSize);
	
	/* if there is another update coming, it gets added in to existing 
	 * context; if the next step is a final, the current digest state is used. */
	return errSecSuccess;
}

OSStatus hmacLegacyFinal(
	hmacLegacyContextRef hmac,
	void *resultPtr)		// caller mallocs, must be HMACSHA1_OUT_SIZE bytes
{
	memcpy (resultPtr, sha1Digest (hmac->sha1Context), kSHA1DigestSize);
	return errSecSuccess;
}

