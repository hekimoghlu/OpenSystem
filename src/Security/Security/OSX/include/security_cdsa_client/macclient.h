/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
// macclient - client interface to CSSM sign/verify mac contexts
//
#ifndef _H_CDSA_CLIENT_MACCLIENT
#define _H_CDSA_CLIENT_MACCLIENT  1

#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_client/keyclient.h>

namespace Security
{

namespace CssmClient
{

//
// A signing/verifying mac context
//
class MacContext : public Context
{
public:
	MacContext(const CSP &csp, CSSM_ALGORITHMS alg)
		: Context(csp, alg) { }

	// preliminary interface
	Key key() const { assert(mKey); return mKey; }
	void key(const Key &k) { mKey = k; set(CSSM_ATTRIBUTE_KEY, mKey); }

protected:
	void activate();
	Key mKey;
};


class GenerateMac : public MacContext
{
public:
	GenerateMac(const CSP &csp, CSSM_ALGORITHMS alg) : MacContext(csp, alg) { }
	
	// integrated
	void sign(const CssmData &data, CssmData &mac) { sign(&data, 1, mac); }
	void sign(const CssmData *data, uint32 count, CssmData &mac);
	
	// staged
	void init(); // Optional
	void sign(const CssmData &data) { sign(&data, 1); }
	void sign(const CssmData *data, uint32 count);
	void operator () (CssmData &mac);
	CssmData operator () () { CssmData mac; (*this)(mac); return mac; }
};

class VerifyMac : public MacContext
{
public:
	VerifyMac(const CSP &csp, CSSM_ALGORITHMS alg) : MacContext(csp, alg) { }
	
	// integrated
	void verify(const CssmData &data, const CssmData &mac) { verify(&data, 1, mac); }
	void verify(const CssmData *data, uint32 count, const CssmData &mac);
	
	// staged
	void init(); // Optional
	void verify(const CssmData &data) { verify(&data, 1); }
	void verify(const CssmData *data, uint32 count);
	void operator () (const CssmData &mac);
};

} // end namespace CssmClient

} // end namespace Security

#endif // _H_CDSA_CLIENT_MACCLIENT
