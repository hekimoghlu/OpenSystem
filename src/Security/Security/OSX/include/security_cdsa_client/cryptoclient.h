/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
// cryptoclient - client interface to CSSM CSP encryption/decryption operations
//
#ifndef _H_CDSA_CLIENT_CRYPTOCLIENT
#define _H_CDSA_CLIENT_CRYPTOCLIENT  1

#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_client/keyclient.h>

namespace Security {
namespace CssmClient {


//
// Common features of various cryptographic operations contexts.
// These all use symmetric or asymmetric contexts.
//
class Crypt : public Context {
public:
	Crypt(const CSP &csp, CSSM_ALGORITHMS alg);
	
public:
    // Context attributes
	CSSM_ENCRYPT_MODE mode() const			{ return mMode; }
	void mode(CSSM_ENCRYPT_MODE m)			{ mMode = m; set(CSSM_ATTRIBUTE_MODE, m); }
	Key key() const							{ return mKey; }
	void key(const Key &k);
	const CssmData &initVector() const		{ return *mInitVector; }
    // The following function is invalid: you cannot save a pointer to an object passed in by reference.
    // Fixing this error leads to corrupted mutexes everywhere; I cannot figure out why.
    // To use the Crypt class, you must ensure that the CssmData object you pass in here lives for the lifetime of Crypt.
	void initVector(const CssmData &v)		{ mInitVector = &v; set(CSSM_ATTRIBUTE_INIT_VECTOR, v); }
	CSSM_PADDING padding() const			{ return mPadding; }
	void padding(CSSM_PADDING p)			{ mPadding = p; set(CSSM_ATTRIBUTE_PADDING, p); }

protected:
	void activate();
	
protected:
	CSSM_ENCRYPT_MODE mMode;
	Key mKey;
	const CssmData *mInitVector;
	CSSM_PADDING mPadding;
    RecursiveMutex mActivateMutex;
};



//
// An encryption context
//
class Encrypt : public Crypt
{
public:
	Encrypt(const CSP &csp, CSSM_ALGORITHMS alg) : Crypt(csp, alg) {};
	
public:
	// integrated
	CSSM_SIZE encrypt(const CssmData *in, uint32 inCount, CssmData *out, uint32 outCount,
		CssmData &remData);
	CSSM_SIZE encrypt(const CssmData &in, CssmData &out, CssmData &remData)
	{ return encrypt(&in, 1, &out, 1, remData); }
	
	// staged update
	void init(); // Optional
	CSSM_SIZE encrypt(const CssmData *in, uint32 inCount, CssmData *out, uint32 outCount);
	CSSM_SIZE encrypt(const CssmData &in, CssmData &out)
	{ return encrypt(&in, 1, &out, 1); }
	// staged final
	void final(CssmData &remData);
};

//
// An Decryption context
//
class Decrypt : public Crypt
{
public:
	Decrypt(const CSP &csp, CSSM_ALGORITHMS alg) : Crypt(csp, alg) {};
	
public:
	// integrated
	CSSM_SIZE decrypt(const CssmData *in, uint32 inCount, CssmData *out, uint32 outCount,
		CssmData &remData);
	CSSM_SIZE decrypt(const CssmData &in, CssmData &out, CssmData &remData)
	{ return decrypt(&in, 1, &out, 1, remData); }

	// staged update
	void init(); // Optional
	CSSM_SIZE decrypt(const CssmData *in, uint32 inCount, CssmData *out, uint32 outCount);
	CSSM_SIZE decrypt(const CssmData &in, CssmData &out)
	{ return decrypt(&in, 1, &out, 1); }
	// staged final
	void final(CssmData &remData);
};


} // end namespace CssmClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_CRYPTOCLIENT
