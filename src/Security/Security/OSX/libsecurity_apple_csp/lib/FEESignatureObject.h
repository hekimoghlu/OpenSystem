/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
 * FEESignatureObject.h - FEE-based raw sign/verify classes
 */

#ifdef	CRYPTKIT_CSP_ENABLE

#ifndef	_FEE_SIGNATURE_OBJECT_H_
#define _FEE_SIGNATURE_OBJECT_H_

#include <security_cryptkit/feePublicKey.h>
#include <security_cryptkit/feeECDSA.h>
#include "FEECSPUtils.h"
#include "CryptKitSpace.h"
#include <RawSigner.h>
#include <AppleCSPSession.h>

namespace CryptKit {

/* 
 * Common raw FEE sign/verify class.
 */
class FEESigner : public RawSigner {
public:
	FEESigner(
		feeRandFcn		randFcn,
		void			*randRef,
		AppleCSPSession &session,
		Allocator	&alloc) : 
			RawSigner(alloc, CSSM_ALGID_NONE),
			mFeeKey(NULL),
			mWeMallocdFeeKey(false),
			mRandFcn(randFcn),
			mRandRef(randRef),
			mSession(session) { }
			
	virtual ~FEESigner();
	
	/* reusable init */
	void signerInit(
		const Context 	&context,
		bool			isSigning);
	
	/* 
	 * obtain key from context, validate, convert to native FEE key
	 */
	void keyFromContext(
		const Context 	&context);

    /*
     * obtain signature format from context
     */
    void sigFormatFromContext(
        const Context 	&context);

protected:
        feeSigFormat    mSigFormat;
        feePubKey		mFeeKey;
		bool			mWeMallocdFeeKey;
		feeRandFcn		mRandFcn;
		void			*mRandRef;
		AppleCSPSession	&mSession;
};

/* 
 * And two implementations.
 *
 * Native FEE signature, ElGamal style.
 */
class FEERawSigner : public FEESigner
{
public:
	FEERawSigner(
		feeRandFcn		randFcn,
		void			*randRef,
		AppleCSPSession &session,
		Allocator	&alloc) : 
			FEESigner(randFcn, randRef, session, alloc) { };
			
	~FEERawSigner() { }
	
	/* sign */
	void sign(
		const void	 	*data, 
		size_t 			dataLen,
		void			*sig,	
		size_t			*sigLen);	/* IN/OUT */
		
	/* verify */
	void verify(
		const void 		*data, 
		size_t 			dataLen,
		const void		*sig,			
		size_t			sigLen);	
		
	/* works for both, but only used for signing */
	size_t maxSigSize();
};

/*
 * FEE signature, ECDSA style.
 */
class FEEECDSASigner : public FEESigner
{
public:
	FEEECDSASigner(
		feeRandFcn		randFcn,
		void			*randRef,
		AppleCSPSession &session,
		Allocator	&alloc) : 
			FEESigner(randFcn, randRef, session, alloc) { };
			
	~FEEECDSASigner() { }
	
	/* sign */
	void sign(
		const void	 	*data, 
		size_t 			dataLen,
		void			*sig,	
		size_t			*sigLen);	/* IN/OUT */
		
	/* verify */
	void verify(
		const void	 	*data, 
		size_t 			dataLen,
		const void		*sig,			
		size_t			sigLen);	
		
	/* works for both, but only used for signing */
	size_t maxSigSize();
};

} /* namespace CryptKit */

#endif	/* _FEE_SIGNATURE_OBJECT_H_ */
#endif	/* CRYPTKIT_CSP_ENABLE */
