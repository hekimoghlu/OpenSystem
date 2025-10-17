/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
 * RawSigner.h - low-level virtual sign/verify object (no digest)
 */
 
#ifndef	_RAW_SIGNER_H_
#define _RAW_SIGNER_H_

#include <security_cdsa_utilities/context.h>
#include <security_utilities/alloc.h>

class RawSigner {
public:
	RawSigner(
		Allocator	&alloc,
		CSSM_ALGORITHMS	digestAlg)	:
			mInitFlag(false), 
			mIsSigning(false),
			mDigestAlg(digestAlg),
			mAlloc(alloc) { }
	virtual ~RawSigner()  	{ }
	
	/* 
	 * The use of our mDigestAlg variable is pretty crufty. For some algs, it's 
	 * known and specified at construction time (e.g., CSSM_ALGID_MD5WithRSA). 
	 * For some algs, it's set by CSPFullPluginSession via 
	 * CSPContext::setDigestAlgorithm during raw sign/verify.
	 */
	void 			setDigestAlg(CSSM_ALGORITHMS alg)
											{ mDigestAlg = alg; }

	/* 
	 * The remaining functions must be implemented by subclass. 
	 */

	/* reusable init */
	virtual void signerInit(
		const Context 	&context,
		bool			isSigning) = 0;
	
	/* sign */
	virtual void sign(
		const void 		*data, 
		size_t 			dataLen,
		void			*sig,	
		size_t			*sigLen) = 0;	/* IN/OUT */
		
	/* verify */
	virtual void verify(
		const void 		*data, 
		size_t 			dataLen,
		const void		*sig,			
		size_t			sigLen) = 0;	
		
	/* works for both, but only used for signing */
	virtual size_t maxSigSize() = 0;

protected:
	bool			mInitFlag;				// true after init
	bool			mOpStarted;				// true after update
	bool			mIsSigning;
	CSSM_ALGORITHMS	mDigestAlg;				// for raw sign/verify
	Allocator	&mAlloc;
	
	bool			initFlag() 				{ return mInitFlag; }
	void			setInitFlag(bool flag) 	{ mInitFlag = flag; }
	bool			opStarted() 			{ return mOpStarted; }
	void			setOpStarted(bool flag) { mOpStarted = flag; }
	bool			isSigning()				{ return mIsSigning; }
	void			setIsSigning(bool signing)
											{ mIsSigning = signing; }
	CSSM_ALGORITHMS	digestAlg()				{ return mDigestAlg; }
	Allocator	&alloc()				{ return mAlloc; }
};


#endif	/* _RAW_SIGNER_H_ */
