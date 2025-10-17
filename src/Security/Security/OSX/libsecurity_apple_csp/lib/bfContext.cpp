/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
 * bfContext.cpp - glue between BlockCrytpor and ssleay Blowfish
 *				 implementation
 */
 
#include "bfContext.h"

BlowfishContext::~BlowfishContext()
{
	deleteKey();
}

void BlowfishContext::deleteKey()
{
	memset(&mBfKey, 0, sizeof(mBfKey));
	mInitFlag = false;
}

/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void BlowfishContext::init( 
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
	symmetricKeyBits(context, session(), CSSM_ALGID_BLOWFISH, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	if((keyLen < BF_MIN_KEY_SIZE_BYTES) || (keyLen > BF_MAX_KEY_SIZE_BYTES)) {
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
		BF_set_key(&mBfKey, (int)keyLen, keyData);
	
		/* save this raw key data */
		memmove(mRawKey, keyData, keyLen); 
		mRawKeySize = (unsigned int)keyLen;
	}
	
	/* Finally, have BlockCryptor do its setup */
	setup(BF_BLOCK, context);
	mInitFlag = true;
}	

/*
 * Functions called by BlockCryptor
 */
void BlowfishContext::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != BF_BLOCK) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < BF_BLOCK) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	BF_ecb_encrypt((const unsigned char *)plainText, (unsigned char *)cipherText,
		&mBfKey, BF_ENCRYPT);
	cipherTextLen = BF_BLOCK;
}

void BlowfishContext::decryptBlock(
	const void		*cipherText,		// length implied (one block)
	size_t			cipherTextLen,
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < BF_BLOCK) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	BF_ecb_encrypt((const unsigned char *)cipherText, (unsigned char *)plainText,
		&mBfKey, BF_DECRYPT);
	plainTextLen = BF_BLOCK;
}
