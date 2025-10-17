/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
 * SignatureContext.h - AppleCSPContext sublass for generic sign/verify
 */

#include "SignatureContext.h"
#include "AppleCSPUtils.h"
#include "AppleCSPSession.h"
#include <Security/cssmtype.h>

#include <security_utilities/debugging.h>

#define cspSigDebug(args...)	secinfo("cspSig", ## args)

SignatureContext::~SignatureContext()
{
	delete &mDigest;
	delete &mSigner;
	mInitFlag = false;
}

/* both sign & verify */
void SignatureContext::init(
	const Context &context, 
	bool isSigning)
{
	mDigest.digestInit();
	mSigner.signerInit(context, isSigning);
	mInitFlag = true;
}

/* both sign & verify */
void SignatureContext::update(
	const CssmData &data)
{
	mDigest.digestUpdate(data.Data, data.Length);
}

/* sign only */
void SignatureContext::final(
	CssmData &out)
{	
	void 		*digest;
	size_t		digestLen;
	void		*sig = out.data();
	size_t		sigLen = out.length();
	
	/* first obtain the digest */
	digestLen = mDigest.digestSizeInBytes();
	digest = session().malloc(digestLen);
	mDigest.digestFinal(digest);
	
	/* now sign */
	try {
		mSigner.sign(digest, 
			digestLen,
			sig,
			&sigLen);
	}
	catch(...) {
		session().free(digest);
		throw;
	}
	session().free(digest);
	if(out.length() < sigLen) {
		cspSigDebug("SignatureContext: mallocd sig too small!");
		CssmError::throwMe(CSSMERR_CSP_INTERNAL_ERROR);
	}
	out.length(sigLen);
}

/* verify only */
void SignatureContext::final(
	const CssmData &in)
{	
	void 		*digest;
	size_t		digestLen;
	
	/* first obtain the digest */
	digestLen = mDigest.digestSizeInBytes();
	digest = session().malloc(digestLen);
	mDigest.digestFinal(digest);
	
	/* now verify */
	try {
		mSigner.verify(digest, 
			digestLen,
			in.Data,
			in.Length);
	}
	catch(...) {
		session().free(digest);
		throw;
	}
	session().free(digest);
}

size_t SignatureContext::outputSize(
	bool final,
	size_t inSize)
{
	return mSigner.maxSigSize();
}

/* for raw sign/verify - optionally called after init */ 
void SignatureContext::setDigestAlgorithm(
	CSSM_ALGORITHMS digestAlg)
{
	mSigner.setDigestAlg(digestAlg);
}
