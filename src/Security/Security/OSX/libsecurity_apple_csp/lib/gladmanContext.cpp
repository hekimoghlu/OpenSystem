/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
 * gladmanContext.cpp - glue between BlockCryptor and Gladman AES implementation
 */
 
#include "gladmanContext.h"
#include "cspdebugging.h"
#include <CommonCrypto/CommonCryptor.h>

/*
 * AES encrypt/decrypt.
 */
GAESContext::GAESContext(AppleCSPSession &session) : 
    BlockCryptor(session),
	mAesKey(NULL),
	mInitFlag(false),
	mRawKeySize(0),
	mWasEncrypting(false)
{ 
	cbcCapable(false);
	multiBlockCapable(false);
}

GAESContext::~GAESContext()
{
    if(mAesKey) {
        CCCryptorFinal(mAesKey,NULL,0,NULL);
        CCCryptorRelease(mAesKey);
        mAesKey = NULL;
    }
    
	deleteKey();
	memset(mRawKey, 0, MAX_AES_KEY_BITS / 8);
	mInitFlag = false;
}
	
void GAESContext::deleteKey()
{
	mRawKeySize = 0;
}

/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt. Even reusable after context
 * changed (i.e., new IV in Encrypted File System). 
 */
void GAESContext::init( 
	const Context &context, 
	bool encrypting)
{
	if(mInitFlag && !opStarted()) {
		return;
	}
	
	CSSM_SIZE	keyLen;
	uint8 		*keyData = NULL;
	bool		sameKeySize = false;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_AES, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	
	switch(keyLen) {
		case kCCKeySizeAES128:
		case kCCKeySizeAES192:
		case kCCKeySizeAES256:
			break;
		default:
			CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	
	/*
	 * Delete existing key if key size changed
	 */
	if(mRawKeySize == keyLen) {
		sameKeySize = true;
	}
	else {
		deleteKey();
	}
	
	/* 
	 * Init key only if key size or key bits have changed, or 
	 * we're doing a different operation than the previous key
	 * was scheduled for.
	 */
	if(!sameKeySize || (mWasEncrypting != encrypting) ||
		memcmp(mRawKey, keyData, mRawKeySize)) {
        (void) CCCryptorCreateWithMode(0, kCCModeECB, kCCAlgorithmAES128, ccDefaultPadding, NULL, keyData, keyLen, NULL, 0, 0, 0, &mAesKey);

		/* save this raw key data */
		memmove(mRawKey, keyData, keyLen); 
		mRawKeySize = (uint32)keyLen;
		mWasEncrypting = encrypting;
	}

	/* we handle CBC, and hence the IV, ourselves */
	CSSM_ENCRYPT_MODE cssmMode = context.getInt(CSSM_ATTRIBUTE_MODE);
    switch (cssmMode) {
		/* no mode attr --> 0 == CSSM_ALGMODE_NONE, not currently supported */
 		case CSSM_ALGMODE_CBCPadIV8:
		case CSSM_ALGMODE_CBC_IV8:
		{
			CssmData *iv = context.get<CssmData>(CSSM_ATTRIBUTE_INIT_VECTOR);
			if(iv == NULL) {
				CssmError::throwMe(CSSMERR_CSP_MISSING_ATTR_INIT_VECTOR);
			}
			if(iv->Length != kCCBlockSizeAES128) {
				CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_INIT_VECTOR);
			}
		}
		break;
		default:
		break;
	}
	
	/* Finally, have BlockCryptor do its setup */
	setup(GLADMAN_BLOCK_SIZE_BYTES, context);
	mInitFlag = true;
}	

/*
 * Functions called by BlockCryptor
 * FIXME make this multi-block capabl3e 
 */
void GAESContext::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void 			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(cipherTextLen < plainTextLen) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorEncryptDataBlock(mAesKey, NULL, plainText, plainTextLen, cipherText);

	cipherTextLen = plainTextLen;
}

void GAESContext::decryptBlock(
	const void		*cipherText,		// length implied (one cipher block)
	size_t			cipherTextLen,	
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < cipherTextLen) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorDecryptDataBlock(mAesKey, NULL, cipherText, cipherTextLen, plainText);
	plainTextLen = cipherTextLen;
}

