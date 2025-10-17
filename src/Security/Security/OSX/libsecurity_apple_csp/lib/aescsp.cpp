/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
 * aescsp.cpp - glue between BlockCryptor and AES implementation
 */
 
#include "aescspi.h"
#include "rijndaelApi.h"
#include "rijndael-alg-ref.h"
#include "cspdebugging.h"

#define DEFAULT_BLOCK_SIZE		(MIN_AES_BLOCK_BITS / 8)

/*
 * AES symmetric key generation.
 * This algorithm has key size restrictions which don't fit with the 
 * standard AppleSymmKeyGenContext model so we have to do some addditional
 * checking.
 */
void AESKeyGenContext::generate(
	const Context 	&context, 
	CssmKey 		&symKey, 
	CssmKey 		&dummyKey)
{
	uint32 reqKeySize = context.getInt(
		CSSM_ATTRIBUTE_KEY_LENGTH, 
		CSSMERR_CSP_MISSING_ATTR_KEY_LENGTH);
	switch(reqKeySize) {
		case MIN_AES_KEY_BITS:
		case MID_AES_KEY_BITS:
		case MAX_AES_KEY_BITS:
			break;
		default:
			CssmError::throwMe(CSSMERR_CSP_UNSUPPORTED_KEY_SIZE);
	}
	AppleSymmKeyGenContext::generateSymKey(
		context, 
		session(),
		symKey);		
}

/*
 * AES encrypt/decrypt.
 */
AESContext::~AESContext()
{
	deleteKey();
	memset(mRawKey, 0, MAX_AES_KEY_BITS / 8);
	mInitFlag = false;
}
	
void AESContext::aesError(
	int artn, 
	const char *errStr)
{
	CSSM_RETURN crtn;
	errorLog2("AESContext: %s : %d\n", errStr, artn);
	switch(artn) {
		case BAD_KEY_INSTANCE:
		default:
			crtn = CSSMERR_CSP_INTERNAL_ERROR;
			break;
		case BAD_KEY_MAT:
			crtn = CSSMERR_CSP_INVALID_KEY;
			break;
	}
	CssmError::throwMe(crtn);
}

void AESContext::deleteKey()
{
	if(mAesKey) {
		memset(mAesKey, 0, sizeof(keyInstance));
		session().free(mAesKey);
		mAesKey = NULL;
	}
}

/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt. Even reusable after context
 * changed (i.e., new IV in Encrypted File System). 
 */
void AESContext::init( 
	const Context &context, 
	bool encrypting)
{
	if(mInitFlag && !opStarted()) {
		return;
	}
	
	CSSM_SIZE	keyLen;
	uint8 		*keyData = NULL;
	unsigned	lastBlockSize = mBlockSize;		// may be 0 (first time thru)
	bool		sameKeyAndBlockSizes = false;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_AES, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	
	switch(keyLen) {
		case MIN_AES_KEY_BITS / 8:
		case MID_AES_KEY_BITS / 8:
		case MAX_AES_KEY_BITS / 8:
			break;
		default:
			CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}

	/* 
	 * Validate context 
	 * block size is optional 
	 */
	mBlockSize = context.getInt(CSSM_ATTRIBUTE_BLOCK_SIZE);
	if(mBlockSize == 0) {
		mBlockSize = DEFAULT_BLOCK_SIZE;		
	}
		
	
	/*
	 * Delete existing key if key size or block size changed
	 */
	if((lastBlockSize == mBlockSize) && (mRawKeySize == keyLen)) {
		sameKeyAndBlockSizes = true;
	}
	if((mAesKey != NULL) && !sameKeyAndBlockSizes) {
		deleteKey();
	}
	
	int opt128 = 0;
#if		!GLADMAN_AES_128_ENABLE
	if((mBlockSize == (MIN_AES_BLOCK_BITS/8)) &&
	   (keyLen == (MIN_AES_KEY_BITS/8)) &&
	   doAES128) {
		opt128 = 1;
	}
#endif	/* !GLADMAN_AES_128_ENABLE */
	
	/* create new key if needed */
	if(mAesKey == NULL) {
		mAesKey = (keyInstance *)session().malloc(sizeof(keyInstance));
	}
	
	/* init key only if key size, block size, or key bits have changed */
	if(!sameKeyAndBlockSizes || memcmp(mRawKey, keyData, mRawKeySize)) {
		int artn = makeKey((keyInstance *)mAesKey, 
			(int)keyLen * 8,
                        mBlockSize * 8,
			(word8 *)keyData,
			opt128);
		if(artn < 0) {
			aesError(artn, "makeKey");
		}
		
		/* save this raw key data */
		memmove(mRawKey, keyData, mRawKeySize); 
		mRawKeySize = (uint32)keyLen;
	}

#if		!GLADMAN_AES_128_ENABLE
	if(opt128) {
		/* optimized path */
		mEncryptFcn = rijndaelBlockEncrypt128;
		mDecryptFcn = rijndaelBlockDecrypt128;
	}
	else {
		/* common standard path */
		mEncryptFcn = rijndaelBlockEncrypt;
		mDecryptFcn = rijndaelBlockDecrypt;
	}
#else
	/* common standard path */
	mEncryptFcn = rijndaelBlockEncrypt;
	mDecryptFcn = rijndaelBlockDecrypt;
#endif /* !GLADMAN_AES_128_ENABLE */
	
	/* Finally, have BlockCryptor do its setup */
	setup(mBlockSize, context);
	mInitFlag = true;
}	

/*
 * Functions called by BlockCryptor
 */
void AESContext::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void 			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != mBlockSize) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < mBlockSize) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	int artn = mEncryptFcn(mAesKey, 
		(word8 *)plainText, 
		(word8 *)cipherText);
	if(artn < 0) {
		aesError(artn, "rijndaelBlockEncrypt");
	}
	cipherTextLen = mBlockSize;
}

void AESContext::decryptBlock(
	const void		*cipherText,		// length implied (one cipher block)
	size_t			cipherTextLen,	
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < mBlockSize) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	int artn = mDecryptFcn(mAesKey, 
		(word8 *)cipherText, 
		(word8 *)plainText);
	if(artn < 0) {
		aesError(artn, "rijndaelBlockDecrypt");
	}
	plainTextLen = mBlockSize;
}

