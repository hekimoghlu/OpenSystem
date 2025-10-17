/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
 * rc4Context.cpp - glue between AppleCSPContext and ssleay RC4 implementation
 */
 
#include "rc4Context.h"

RC4Context::~RC4Context()
{
    if (rc4Key != NULL) {
        CCCryptorRelease(rc4Key);
    }
    rc4Key = NULL;
}
	
/* 
 * Standard CSPContext init, called from CSPFullPluginSession::init().
 * Reusable, e.g., query followed by en/decrypt.
 */
void RC4Context::init( 
	const Context &context, 
	bool encrypting)
{
	CSSM_SIZE	keyLen;
	uint8 		*keyData 	= NULL;
	
	/* obtain key from context */
	symmetricKeyBits(context, session(), CSSM_ALGID_RC4, 
		encrypting ? CSSM_KEYUSE_ENCRYPT : CSSM_KEYUSE_DECRYPT,
		keyData, keyLen);
	if((keyLen < kCCKeySizeMinRC4) || (keyLen > kCCKeySizeMaxRC4)) {
		CssmError::throwMe(CSSMERR_CSP_INVALID_ATTR_KEY);
	}
	
	/* All other context attributes ignored */
	/* init the low-level state */
    (void) CCCryptorCreateWithMode(0, kCCModeRC4, kCCAlgorithmRC4, ccDefaultPadding, NULL, keyData, keyLen, NULL, 0, 0, 0, &rc4Key);

}	

/*
 * All of these functions are called by CSPFullPluginSession.
 */
void RC4Context::update(
	void 			*inp, 
	size_t 			&inSize, 			// in/out
	void 			*outp, 
	size_t 			&outSize)			// in/out
{
    (void) CCCryptorUpdate(rc4Key, inp, inSize, outp, inSize, &outSize);
}

/* remainding functions are trivial for any stream cipher */
void RC4Context::final(
	CssmData 		&out)	
{
	out.length(0);
}

size_t RC4Context::inputSize(
	size_t 			outSize)			// input for given output size
{
	return outSize;
}

size_t RC4Context::outputSize(
	bool 			final /*= false*/, 
	size_t 			inSize /*= 0*/) 	// output for given input size
{
	return inSize;
}

void RC4Context::minimumProgress(
	size_t 			&in, 
	size_t 			&out) 				// minimum progress chunks
{
	in  = 1;
	out = 1;
}
