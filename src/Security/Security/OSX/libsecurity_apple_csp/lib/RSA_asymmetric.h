/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
 * RSA_asymmetric.h - CSPContext for RSA asymmetric encryption
 */

#ifndef _RSA_ASYMMETRIC_H_
#define _RSA_ASYMMETRIC_H_

#include <security_cdsa_plugin/CSPsession.h>
#include <AppleCSP.h>
#include <AppleCSPContext.h>
#include <AppleCSPSession.h>
#include <BlockCryptor.h>
#include <openssl/rsa_legacy.h>

#define RSA_ASYM_PADDING_DEFAULT		RSA_PKCS1_PADDING

class RSA_CryptContext : public BlockCryptor {
public:
	RSA_CryptContext(AppleCSPSession &session) :
		BlockCryptor(session),
		mRsaKey(NULL),
		mAllocdRsaKey(false),
		mInitFlag(false),
		mPadding(RSA_ASYM_PADDING_DEFAULT),
		mOaep(false),
		mLabel(Allocator::standard()) 	{ }
		
	~RSA_CryptContext();
	
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

	size_t outputSize(
		bool 			final,				// ignored
		size_t 			inSize = 0); 		// output for given input size

private:
		RSA				*mRsaKey;
		bool			mAllocdRsaKey;
		bool			mInitFlag;			// allows easy reuse
		int				mPadding;			// RSA_NO_PADDING, RSA_PKCS1_PADDING,
											//    RSA_SSLV23_PADDING

		/* 
		 * optional fields for OEAP keys 
		 * (mKeyHeader.AlgorithmId == CSSM_ALGMODE_PKCS1_EME_OAEP) 
		 */
		bool					mOaep;
		CssmAutoData			mLabel;
		
};	/* RSA_CryptContext */


#endif // _RSA_ASYMMETRIC_H_
