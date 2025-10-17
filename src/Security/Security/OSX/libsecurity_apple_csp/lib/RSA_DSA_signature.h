/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
 * RSA_DSA_signature.h - openssl-based signature classes.  
 */

#ifndef	_RSA_DSA_SIGNATURE_H_
#define _RSA_DSA_SIGNATURE_H_

#include <openssl/rsa_legacy.h>
#include <openssl/dsa_legacy.h>
#include <RawSigner.h>
#include <AppleCSPSession.h>

#define RSA_SIG_PADDING_DEFAULT		RSA_PKCS1_PADDING

class RSASigner : public RawSigner {
public:
	RSASigner(
		Allocator	&alloc,
		AppleCSPSession &session,
		CSSM_ALGORITHMS	digestAlg) : 
			RawSigner(alloc, digestAlg),
			mRsaKey(NULL),
			mWeMallocdRsaKey(false),
			mSession(session),
			mPadding(RSA_SIG_PADDING_DEFAULT) { }
			
	~RSASigner();
	
	/* reusable init */
	void signerInit(
		const Context 	&context,
		bool			isSigning);
	

	/* sign */
	void sign(
		const void 		*data, 
		size_t 			dataLen,
		void			*sig,	
		size_t			*sigLen);	/* IN/OUT */
		
	/* verify */
	void verify(
		const void 	*data, 
		size_t 			dataLen,
		const void		*sig,			
		size_t			sigLen);	
		
	/* works for both, but only used for signing */
	size_t maxSigSize();

private:

	/* 
	 * obtain key from context, validate, convert to RSA key
	 */
	void keyFromContext(
		const Context 	&context);

	RSA					*mRsaKey;
	bool				mWeMallocdRsaKey;
	AppleCSPSession		&mSession;
	int					mPadding;		// RSA_NO_PADDING, RSA_PKCS1_PADDING
};

class DSASigner : public RawSigner {
public:
	DSASigner(
		Allocator	&alloc,
		AppleCSPSession &session,
		CSSM_ALGORITHMS	digestAlg) : 
			RawSigner(alloc, digestAlg),
			mDsaKey(NULL),
			mWeMallocdDsaKey(false),
			mSession(session) { }
			
	~DSASigner();
	
	/* reusable init */
	void signerInit(
		const Context 	&context,
		bool			isSigning);
	

	/* sign */
	void sign(
		const void 		*data, 
		size_t 			dataLen,
		void			*sig,	
		size_t			*sigLen);	/* IN/OUT */
		
	/* verify */
	void verify(
		const void 	*data, 
		size_t 			dataLen,
		const void		*sig,			
		size_t			sigLen);	
		
	/* works for both, but only used for signing */
	size_t maxSigSize();

private:

	/* 
	 * obtain key from context, validate, convert to DSA key
	 */
	void keyFromContext(
		const Context 	&context);

	DSA					*mDsaKey;
	bool				mWeMallocdDsaKey;
	AppleCSPSession		&mSession;
};


#endif	/* _RSA_DSA_SIGNATURE_H_ */
