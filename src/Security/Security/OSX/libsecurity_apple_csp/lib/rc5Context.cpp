/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
 * rc5Context.cpp - glue between BlockCrytpor and ssleay RC5 implementation
 */
 
#include <openssl/rc5_legacy.h>
#include <misc/rc5_locl.h>
#include "rc5Context.h"

RC5Context::~RC5Context()
{
	memset(&rc5Key, 0, sizeof(RC5_32_KEY));
}
	
/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void RC5Context::init( 
	const Context &context, 
	bool encrypting)
{
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	uint32		rounds = RC5_16_ROUNDS;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_RC5, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	if((keyLen < RC5_MIN_KEY_SIZE_BYTES) || (keyLen > RC5_MAX_KEY_SIZE_BYTES)) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	
	/* 
	 * Optional rounds
	 */
	rounds = context.getInt(CSSM_ATTRIBUTE_ROUNDS);
	if(rounds == 0) {
		/* default */
		rounds = RC5_16_ROUNDS;
	}

	/* init the low-level state */
	RC5_32_set_key(&rc5Key, (int)keyLen, keyData, rounds);

	/* Finally, have BlockCryptor do its setup */
	setup(RC5_BLOCK_SIZE_BYTES, context);
}	

/*
 * Functions called by BlockCryptor
 */
void RC5Context::encryptBlock(
	const void		*plainText,			// length implied (one block)
	size_t			plainTextLen,
	void			*cipherText,	
	size_t			&cipherTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen != RC5_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_INPUT_LENGTH_ERROR);
	}
	if(cipherTextLen < RC5_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	
	/*
	 * Low-level code operates on array of unsigned 32-bit integers 
	 */
	RC5_32_INT	d[2];
	RC5_32_INT l;
	const unsigned char *pt = (const unsigned char *)plainText;
	c2l(pt, l); d[0]=l;
	c2l(pt, l); d[1]=l;
	RC5_32_encrypt(d, &rc5Key);
	unsigned char *ct = (unsigned char *)cipherText;
	l=d[0]; l2c(l, ct);
	l=d[1]; l2c(l, ct);
	cipherTextLen = RC5_BLOCK_SIZE_BYTES;
}

void RC5Context::decryptBlock(
	const void		*cipherText,		// length implied (one block)
	size_t			cipherTextLen,
	void			*plainText,	
	size_t			&plainTextLen,		// in/out, throws on overflow
	bool			final)				// ignored
{
	if(plainTextLen < RC5_BLOCK_SIZE_BYTES) {
		CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
	}
	/*
	 * Low-level code operates on array of unsigned 32-bit integers 
	 */
	RC5_32_INT	d[2];
	RC5_32_INT l;
	const unsigned char *ct = (const unsigned char *)cipherText;
	c2l(ct, l); d[0]=l;
	c2l(ct, l); d[1]=l;
	RC5_32_decrypt(d, &rc5Key);
	unsigned char *pt = (unsigned char *)plainText;
	l=d[0]; l2c(l, pt);
	l=d[1]; l2c(l, pt);
	plainTextLen = RC5_BLOCK_SIZE_BYTES;
}

