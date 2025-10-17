/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
 * castContext.cpp - glue between BlockCrytpor and CommonCrypto CAST128 (CAST5)
 *				 implementation
 *
 */
 
#include "castContext.h"

CastContext::CastContext(AppleCSPSession &session) :
    BlockCryptor(session),
    mCastKey(NULL),
    mInitFlag(false),
    mRawKeySize(0)
{
}



CastContext::~CastContext()
{
 	deleteKey();
}

void CastContext::deleteKey()
{
    if (mCastKey != NULL) {
        CCCryptorRelease(mCastKey);
    }
    mCastKey = NULL;
	mInitFlag = false;
}

/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void CastContext::init( 
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
	symmetricKeyBits(context, session(), CSSM_ALGID_CAST, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	if((keyLen < kCCKeySizeMinCAST) || (keyLen > kCCKeySizeMaxCAST)) {
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

	/* init key only if key size or key bits have changed */
	if(!sameKeySize || memcmp(mRawKey, keyData, mRawKeySize)) {
        (void) CCCryptorCreateWithMode(0, kCCModeECB, kCCAlgorithmCAST, ccDefaultPadding, NULL, keyData, keyLen, NULL, 0, 0, 0, &mCastKey);
	
		/* save this raw key data */
		memmove(mRawKey, keyData, keyLen); 
		mRawKeySize = (uint32)keyLen;
	}
	
	/* Finally, have BlockCryptor do its setup */
	setup(kCCBlockSizeCAST, context);
	mInitFlag = true;
}	

/*
 * Functions called by BlockCryptor
 */
void CastContext::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != kCCBlockSizeCAST) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < kCCBlockSizeCAST) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorEncryptDataBlock(mCastKey, NULL, plainText, kCCBlockSizeCAST, cipherText);

	cipherTextLen = kCCBlockSizeCAST;
}

void CastContext::decryptBlock(
	const void		*cipherText,		// length implied (one block)
	size_t			cipherTextLen,
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < kCCBlockSizeCAST) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorDecryptDataBlock(mCastKey, NULL, cipherText, kCCBlockSizeCAST, plainText);

	plainTextLen = kCCBlockSizeCAST;
}
