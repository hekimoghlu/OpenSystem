/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
 * desContext.cpp - glue between BlockCrytpor and DES implementation
 */
 
#include "desContext.h"
#include <security_utilities/debugging.h>
#include <security_utilities/globalizer.h>
#include <security_utilities/threading.h>

#define DESDebug(args...)	secinfo("desContext", ## args)

/*
 * DES encrypt/decrypt.
 */
DESContext::DESContext(AppleCSPSession &session) : BlockCryptor(session), DesInst(NULL)
{
}

DESContext::~DESContext()
{
    if (DesInst != NULL) {
        CCCryptorRelease(DesInst);
    }
    
    DesInst = NULL;
}
	
/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void DESContext::init( 
	const Context &context, 
	bool encrypting)
{
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_DES, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,	
		keyData, keyLen);
	if(keyLen != (DES_KEY_SIZE_BITS_EXTERNAL / 8)) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	
    if (DesInst != NULL)
    {
        CCCryptorRelease(DesInst);
        DesInst = NULL;
    }
    
    (void) CCCryptorCreateWithMode(0, kCCModeECB, kCCAlgorithmDES, ccDefaultPadding, NULL, keyData, kCCKeySizeDES, NULL, 0, 0, 0, &DesInst);

	/* Finally, have BlockCryptor do its setup */
	setup(DES_BLOCK_SIZE_BYTES, context);
}	

/*
 * Functions called by BlockCryptor
 * DES does encrypt/decrypt in place
 */
void DESContext::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != DES_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < DES_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorEncryptDataBlock(DesInst, NULL, plainText, DES_BLOCK_SIZE_BYTES, cipherText);
	cipherTextLen = DES_BLOCK_SIZE_BYTES;
}

void DESContext::decryptBlock(
	const void		*cipherText,		// length implied (one block)
	size_t			cipherTextLen,
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < DES_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	if(plainText != cipherText) {
		/* little optimization for callers who want to decrypt in place */
		memmove(plainText, cipherText, DES_BLOCK_SIZE_BYTES);
	}
    (void) CCCryptorDecryptDataBlock(DesInst, NULL, cipherText, DES_BLOCK_SIZE_BYTES, plainText);
	plainTextLen = DES_BLOCK_SIZE_BYTES;
}

/***
 *** Triple-DES - EDE, 24-bit key only
 ***/
 
DES3Context::DES3Context(AppleCSPSession &session) : BlockCryptor(session), DesInst(NULL)
{
}



DES3Context::~DES3Context()
{
    if (DesInst != NULL) {
        CCCryptorRelease(DesInst);
    }
    
    DesInst = NULL;
}

/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void DES3Context::init( 
	const Context &context, 
	bool encrypting)
{
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_3DES_3KEY_EDE, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	if(keyLen != DES3_KEY_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}

    if (DesInst != NULL) {
        CCCryptorRelease(DesInst);
        DesInst = NULL;
    }
    
    (void) CCCryptorCreateWithMode(0, kCCModeECB, kCCAlgorithm3DES, ccDefaultPadding, NULL, keyData, kCCKeySize3DES, NULL, 0, 0, 0, &DesInst);

	/* Finally, have BlockCryptor do its setup */
	setup(DES3_BLOCK_SIZE_BYTES, context);
}	

/*
 * Functions called by BlockCryptor
 * DES does encrypt/decrypt in place
 */
void DES3Context::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != DES3_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < DES3_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorEncryptDataBlock(DesInst, NULL, plainText, DES3_BLOCK_SIZE_BYTES, cipherText);
	cipherTextLen = DES3_BLOCK_SIZE_BYTES;
}

void DES3Context::decryptBlock(
	const void		*cipherText,		// length implied (one block)
	size_t			cipherTextLen,
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < DES3_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
    (void) CCCryptorDecryptDataBlock(DesInst, NULL, cipherText, DES3_BLOCK_SIZE_BYTES, plainText);
	plainTextLen = DES3_BLOCK_SIZE_BYTES;
}
