/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#include <security_cdsa_client/genkey.h>

using namespace CssmClient;


GenerateKey::GenerateKey(const CSP &csp, CSSM_ALGORITHMS alg, uint32 size)
: Context(csp, alg), mKeySize(size), mSeed(NULL), mSalt(NULL), mParams(NULL)
{
}

void
GenerateKey::database(const Db &inDb)
{
	mDb = inDb;
	if (mDb && isActive())
		set(CSSM_ATTRIBUTE_DL_DB_HANDLE, mDb->handle());
}

void GenerateKey::activate()
{
    StLock<Mutex> _(mActivateMutex);
	if (!mActive)
	{
		check(CSSM_CSP_CreateKeyGenContext(attachment()->handle(), mAlgorithm,
			mKeySize, mSeed, mSalt, NULL, NULL, mParams, &mHandle));
		// Must be done before calling set() since is does nothing unless we are active.
		// Also we are technically active even if set() throws since we already created a context.
		mActive = true;
		if (mDb)
			set(CSSM_ATTRIBUTE_DL_DB_HANDLE, mDb->handle());
	}
}

Key GenerateKey::operator () (const KeySpec &spec)
{
	Key key;
	
	check(CSSM_GenerateKey(handle(), spec.usage, spec.attributes, spec.label,
		   &compositeRcc(), key.makeNewKey(attachment())));
		   
	key->activate();
	
	return key;
}

void GenerateKey::operator () (CssmKey &key, const KeySpec &spec)
{
	check(CSSM_GenerateKey(handle(), spec.usage, spec.attributes, spec.label, &compositeRcc(), &key));

}

void GenerateKey::operator () (Key &publicKey, const KeySpec &pubSpec,
		Key &privateKey, const KeySpec &privSpec)
{
	check(CSSM_GenerateKeyPair(handle(),
		pubSpec.usage, pubSpec.attributes,
		pubSpec.label, publicKey.makeNewKey(attachment()),
		privSpec.usage, privSpec.attributes,
		privSpec.label, &compositeRcc(), privateKey.makeNewKey(attachment())));

	publicKey->activate();
	privateKey->activate();

}

void GenerateKey::operator () (CssmKey &publicKey, const KeySpec &pubSpec,
		CssmKey &privateKey, const KeySpec &privSpec)
{
	check(CSSM_GenerateKeyPair(handle(),
		pubSpec.usage, pubSpec.attributes, pubSpec.label, &publicKey,
		privSpec.usage, privSpec.attributes, privSpec.label, &compositeRcc(), &privateKey));
}

