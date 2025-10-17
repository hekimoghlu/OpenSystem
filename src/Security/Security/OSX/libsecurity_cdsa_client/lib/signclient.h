/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
//
// signclient - client interface to CSSM sign/verify contexts
//
#ifndef _H_CDSA_CLIENT_SIGNCLIENT
#define _H_CDSA_CLIENT_SIGNCLIENT  1

#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_client/keyclient.h>

namespace Security {
namespace CssmClient {


//
// A signing/verifying context
//
class SigningContext : public Context
{
public:
	SigningContext(const CSP &csp, CSSM_ALGORITHMS alg, CSSM_ALGORITHMS signOnly = CSSM_ALGID_NONE)
	: Context(csp, alg), mSignOnly(signOnly) { }

	Key key() const { assert(mKey); return mKey; }
	void key(const Key &k) { mKey = k; set(CSSM_ATTRIBUTE_KEY, mKey); }
    
    CSSM_ALGORITHMS signOnlyAlgorithm() const	{ return mSignOnly; }
    void signOnlyAlgorithm(CSSM_ALGORITHMS alg)	{ mSignOnly = alg; }

protected:
	void activate();
	CSSM_ALGORITHMS mSignOnly;
	Key mKey;
};


class Sign : public SigningContext
{
public:
	Sign(const CSP &csp, CSSM_ALGORITHMS alg, CSSM_ALGORITHMS signOnly = CSSM_ALGID_NONE)
        : SigningContext(csp, alg, signOnly) { }
	
	// integrated
	void sign(const CssmData &data, CssmData &signature) { sign(&data, 1, signature); }
	void sign(const CssmData *data, uint32 count, CssmData &signature);

	// staged
	void init(); // Optional
	void sign(const CssmData &data) { sign(&data, 1); }
	void sign(const CssmData *data, uint32 count);
	void operator () (CssmData &signature);
	CssmData operator () () { CssmData signature; (*this)(signature); return signature; }
};

class Verify : public SigningContext
{
public:
	Verify(const CSP &csp, CSSM_ALGORITHMS alg, CSSM_ALGORITHMS verifyOnly = CSSM_ALGID_NONE)
        : SigningContext(csp, alg, verifyOnly) { }
	
	// integrated
	void verify(const CssmData &data, const CssmData &signature) { verify(&data, 1, signature); }
	void verify(const CssmData *data, uint32 count, const CssmData &signature);

	// staged
	void init(); // Optional
	void verify(const CssmData &data) { verify(&data, 1); }
	void verify(const CssmData *data, uint32 count);
	void operator () (const CssmData &signature);
};

} // end namespace CssmClient

} // end namespace Security

#endif // _H_CDSA_CLIENT_SIGNCLIENT
