/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
// wrapkey - client interface for wrapping and unwrapping keys
//
#ifndef _H_CDSA_CLIENT_WRAPKEY
#define _H_CDSA_CLIENT_WRAPKEY  1

#include <security_cdsa_client/cspclient.h>
#include <security_cdsa_client/cryptoclient.h>
#include <security_cdsa_client/keyclient.h>


namespace Security {
namespace CssmClient {


//
// Wrap a key
//
class WrapKey : public Crypt {
public:
	WrapKey(const CSP &csp, CSSM_ALGORITHMS alg) :
		Crypt(csp, alg), mWrappedKeyFormat(CSSM_KEYBLOB_WRAPPED_FORMAT_NONE) {}

public:
	CSSM_KEYBLOB_FORMAT wrappedKeyFormat() const { return mWrappedKeyFormat; }
	void wrappedKeyFormat(CSSM_KEYBLOB_FORMAT wrappedKeyFormat)
	{ mWrappedKeyFormat = wrappedKeyFormat; set(CSSM_ATTRIBUTE_WRAPPED_KEY_FORMAT, wrappedKeyFormat); }

	// wrap the key
	Key operator () (Key &keyToBeWrapped, const CssmData *descriptiveData = NULL);
	void operator () (const CssmKey &keyToBeWrapped, CssmKey &wrappedKey,
					  const CssmData *descriptiveData = NULL);

protected:
	void activate();

private:
	CSSM_KEYBLOB_FORMAT mWrappedKeyFormat;
};


//
// Unwrap a key. This creates a new key object
//
class UnwrapKey : public Crypt, public RccBearer {
public:
	UnwrapKey(const CSP &csp, CSSM_ALGORITHMS alg) : Crypt(csp, alg) {}

public:
	// wrap the key
	Key operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec);
	void operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					  CssmKey &unwrappedKey);

	Key operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					 Key &optionalPublicKey);
	void operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					  CssmKey &unwrappedKey, const CssmKey *optionalPublicKey);

	Key operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					 CssmData *descriptiveData);
	void operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					  CssmKey &unwrappedKey, CssmData *descriptiveData);

	Key operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					 const Key &optionalPublicKey, CssmData *descriptiveData);
	void operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
					  CssmKey &unwrappedKey, CssmData *descriptiveData,
					  const CssmKey *optionalPublicKey);
};


//
// Derive a key in various and wonderous ways. Creates a new key object.
//
class DeriveKey : public Crypt, public RccBearer {
public:
	DeriveKey(const CSP &csp, CSSM_ALGORITHMS alg, CSSM_ALGORITHMS target, uint32 size = 0)
    : Crypt(csp, alg), mKeySize(size), mTargetType(target), mIterationCount(0),
      mSeed(NULL), mSalt(NULL) { }

public:
    CSSM_ALGORITHMS targetType() const { return mTargetType; }
    void targetType(CSSM_ALGORITHMS alg) { mTargetType = alg; }
    uint32 iterationCount() const		{ return mIterationCount; }
    void iterationCount(uint32 c)		{ mIterationCount = c; }
    const CssmCryptoData seed() const	{ return *mSeed; }
    void seed(const CssmCryptoData &data) { mSeed = &data; }
    const CssmData salt() const			{ return *mSalt; }
    void salt(const CssmData &data)		{ mSalt = &data; }

	Key operator () (CssmData *param, const KeySpec &spec);
	void operator () (CssmData *param, const KeySpec &spec,
					  CssmKey &derivedKey);
                      
    void activate();
    
private:
	uint32 mKeySize;
    CSSM_ALGORITHMS mTargetType;
    uint32 mIterationCount;
    const CssmCryptoData *mSeed;
    const CssmData *mSalt;
};

} // end namespace CssmClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_WRAPKEY
