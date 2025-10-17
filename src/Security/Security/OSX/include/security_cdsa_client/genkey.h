/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
// genkey - client interface to CSSM sign/verify contexts
//
#ifndef _H_CDSA_CLIENT_GENKEY
#define _H_CDSA_CLIENT_GENKEY  1

#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_client/cryptoclient.h>
#include <security_cdsa_client/dlclient.h>
#include <security_cdsa_client/keyclient.h>


namespace Security
{

namespace CssmClient
{

class GenerateKey : public Context, public RccBearer {
public:
	GenerateKey(const CSP &csp, CSSM_ALGORITHMS alg, uint32 size = 0);

public:
	// context parameters
	void size(uint32 s) { mKeySize = s; set(CSSM_ATTRIBUTE_KEY_LENGTH, s); }
	void seed(const CssmCryptoData &s) { mSeed = &s; set(CSSM_ATTRIBUTE_SEED, s); }
	void salt(const CssmData &s) { mSalt = &s;set(CSSM_ATTRIBUTE_SALT, s);  }
	void params(const CssmData &p) { mParams = &p; set(CSSM_ATTRIBUTE_ALG_PARAMS, p); }
	void database(const Db &inDb);

	// symmetric key generation
	Key operator () (const KeySpec &spec);
	void operator () (CssmKey &key, const KeySpec &spec);
	
	// asymmetric key generation
	void operator () (Key &publicKey, const KeySpec &publicSpec,
		Key &privateKey, const KeySpec &privateSpec);
	void operator () (CssmKey &publicKey, const KeySpec &publicSpec,
		CssmKey &privateKey, const KeySpec &privateSpec);

	
protected:
	void activate();
	
private:
	// context parameters
	uint32 mKeySize;
	const CssmCryptoData *mSeed;
	const CssmData *mSalt;
	const CssmData *mParams;
	Db mDb;

	// generation parameters(?) -- Unused
	// const ResourceControlContext *mInitialAcl;
};

} // end namespace CssmClient

} // end namespace Security

#endif // _H_CDSA_CLIENT_GENKEY
