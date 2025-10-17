/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
 * SignatureContext.h - AppleCSPContext subclass for generic sign/verify
 *
 * This class performs all of the sign/verify operations in the CSP. The general
 * scheme is that an instance of this class has references to one DigestObject
 * and one RawSigner. Sign and Verify "updates" go to the DigestObject. The "final"
 * operation consists of obtaining the final digest from the DigestObject and 
 * performing a sign or verify on that data via the RawSigner. 
 *
 * This class knows nothing about any of the algorithms involved; all sign and 
 * verify operations follow this same scheme. Various modules' AlgorithmFactories
 * construct one of these objects by providing the appropriate DigestObject and
 * RawSigner. 
 *
 * The seemingly special case of "raw RSA sign", in which the app calculates the 
 * digest separately from the sign operation, is handled via the NullDigest object.
 */
 
#ifndef	_SIGNATURE_CONTEXT_H_
#define _SIGNATURE_CONTEXT_H_

#include <RawSigner.h>
#include <security_cdsa_utilities/digestobject.h>
#include <AppleCSPContext.h>

class SignatureContext : public AppleCSPContext  {
public:
	SignatureContext(
		AppleCSPSession &session,
		DigestObject	&digest,
		RawSigner		&signer) : 
			AppleCSPContext(session), 
			mDigest(digest), 
			mSigner(signer),
			mInitFlag(false) { }
			
	~SignatureContext();
	
	/* called out from CSPFullPluginSession....
	 * both sign & verify: */
	void init(const Context &context, bool isSigning);
	void update(const CssmData &data);
	
	/* sign only */
	void final(CssmData &out);	
	
	/* verify only */
	void final(const CssmData &in);	

	size_t outputSize(bool final, size_t inSize);

	/* for raw sign/verify - optionally called after init */ 
	virtual void setDigestAlgorithm(CSSM_ALGORITHMS digestAlg);


private:
	DigestObject	&mDigest;
	RawSigner		&mSigner;
	bool			mInitFlag;			// true after init
};


#endif	/* _SIGNATURE_CONTEXT_H_ */
