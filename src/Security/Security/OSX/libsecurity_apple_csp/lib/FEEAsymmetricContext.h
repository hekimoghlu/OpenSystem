/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
 * FEEAsymmetricContext.h - CSPContexts for FEE asymmetric encryption
 *
 */

#ifdef	CRYPTKIT_CSP_ENABLE

#ifndef _FEE_ASYMMETRIC_CONTEXT_H_
#define _FEE_ASYMMETRIC_CONTEXT_H_

#include <security_cdsa_plugin/CSPsession.h>
#include "AppleCSP.h"
#include "AppleCSPContext.h"
#include "AppleCSPSession.h"
#include "BlockCryptor.h"
#include <security_cryptkit/feeFEED.h>
#include <security_cryptkit/feeFEEDExp.h>

namespace CryptKit {

class FEEDContext : public BlockCryptor {
public:
	FEEDContext(AppleCSPSession &session) :
		BlockCryptor(session),
		mFeeFeed(NULL),
		mPrivKey(NULL),
		mPubKey(NULL),
		mInitFlag(false) 	{ }
	~FEEDContext();
	
	/* called by CSPFullPluginSession */
	void init(const Context &context, bool encoding = true);

	/* called by BlockCryptor */
	void encryptBlock(
		const void		*plainText,			// length implied (one block)
		size_t			plainTextLen,
		void			*cipherText,	
		size_t			&cipherTextLen,		// in/out, throws on overflow
		bool			final);
	void decryptBlock(
		const void		*cipherText,		// length implied (one cipher block)
		size_t			cipherTextLen,
		void			*plainText,	
		size_t			&plainTextLen,		// in/out, throws on overflow
		bool			final);
	
	/*
 	 * Additional query size support, necessary because we don't conform to 
	 * BlockCryptor's standard one-to-one block scheme
	 */
 	size_t inputSize(
		size_t 			outSize);			// input for given output size
	size_t outputSize(
		bool 			final = false, 
		size_t 			inSize = 0); 		// output for given input size
	void minimumProgress(
		size_t 			&in, 
		size_t 			&out); 				// minimum progress chunks


private:
		feeFEED			mFeeFeed;
		feePubKey		mPrivKey;
		bool			mAllocdPrivKey;
		feePubKey		mPubKey;
		bool			mAllocdPubKey;
		bool			mInitFlag;			// allows easy reuse
};	/* FEEDContext */


class FEEDExpContext : public BlockCryptor {
public:
	FEEDExpContext(AppleCSPSession &session) :
		BlockCryptor(session),
		mFeeFeedExp(NULL),
		mFeeKey(NULL),
		mInitFlag(false) 	{ }

	~FEEDExpContext();
	
	/* called by CSPFullPluginSession */
	void init(const Context &context, bool encoding = true);

	/* called by BlockCryptor */
	void encryptBlock(
		const void		*plainText,			// length implied (one block)
		size_t			plainTextLen,
		void			*cipherText,	
		size_t			&cipherTextLen,		// in/out, throws on overflow
		bool			final);
	void decryptBlock(
		const void		*cipherText,		// length implied (one cipher block)
		size_t			cipherTextLen,
		void			*plainText,	
		size_t			&plainTextLen,		// in/out, throws on overflow
		bool			final);
	
private:
		feeFEEDExp		mFeeFeedExp;
		feePubKey		mFeeKey;
		bool			mAllocdFeeKey;
		bool			mInitFlag;			// allows easy reuse
};	/* FEEDExpContext */

/*
 * Elliptic curve Diffie-Hellman key exchange. The public key is 
 * specified in one of two ways - a raw X9.62 format public key 
 * string in Param, or a CSSM_KEY in the Context. 
 * Requested size, in keyData->Length, must be the same size as
 * the keys' modulus. Data is returned in keyData->Data, which is 
 * allocated by the caller.
 * Optionally performs X9.63 key derivation if algId == 
 * CSSM_ALGID_ECDH_X963_KDF, with the optional SharedInfo passed
 * as optional context attribute CSSM_ATTRIBUTE_SALT.
 */
extern void DeriveKey_ECDH (
	const Context &context,
	CSSM_ALGORITHMS algId,		
	const CssmData &Param,
	CSSM_DATA *keyData,
	AppleCSPSession &session);

} /* namespace CryptKit */

#endif 	/* _FEE_ASYMMETRIC_CONTEXT_H_ */
#endif	/* CRYPTKIT_CSP_ENABLE */
