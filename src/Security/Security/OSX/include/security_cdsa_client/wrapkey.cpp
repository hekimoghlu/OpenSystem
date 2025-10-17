/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
#include <security_cdsa_client/wrapkey.h>

namespace Security {
namespace CssmClient {


Key
WrapKey::operator () (Key &keyToBeWrapped, const CssmData *descriptiveData)
{
	Key wrappedKey;

	check(CSSM_WrapKey(handle(), neededCred(), keyToBeWrapped, descriptiveData,
					   wrappedKey.makeNewKey(attachment())));
	wrappedKey->activate();

	return wrappedKey;
}

void
WrapKey::operator () (const CssmKey &keyToBeWrapped, CssmKey &wrappedKey,
					  const CssmData *descriptiveData)
{
	check(CSSM_WrapKey(handle(), neededCred(), &keyToBeWrapped,
		descriptiveData, &wrappedKey));
}

void
WrapKey::activate()
{
	if (!mActive)
	{
		Crypt::activate();
		if (mWrappedKeyFormat != CSSM_KEYBLOB_WRAPPED_FORMAT_NONE)
			set(CSSM_ATTRIBUTE_WRAPPED_KEY_FORMAT, mWrappedKeyFormat);
	}
}

Key
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec)
{
	CssmData data(reinterpret_cast<uint8 *>(1), 0);
	Key unwrappedKey;
	check(CSSM_UnwrapKey(handle(), NULL,
						 &keyToBeUnwrapped, spec.usage, spec.attributes,
						 spec.label, &compositeRcc(),
						 unwrappedKey.makeNewKey(attachment()), &data));
	unwrappedKey->activate();

	return unwrappedKey;
}

void
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						CssmKey &unwrappedKey)
{
	CssmData data(reinterpret_cast<uint8 *>(1), 0);
	check(CSSM_UnwrapKey(handle(), NULL, &keyToBeUnwrapped, spec.usage,
						 spec.attributes, spec.label, &compositeRcc(),
						 &unwrappedKey, &data));
}

Key
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						Key &optionalPublicKey)
{
	CssmData data(reinterpret_cast<uint8 *>(1), 0);
	Key unwrappedKey;
	check(CSSM_UnwrapKey(handle(), optionalPublicKey,
						 &keyToBeUnwrapped, spec.usage, spec.attributes,
						 spec.label, &compositeRcc(),
						 unwrappedKey.makeNewKey(attachment()), &data));

	unwrappedKey->activate();

	return unwrappedKey;
}

void
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						CssmKey &unwrappedKey,
						const CssmKey *optionalPublicKey)
{
	CssmData data(reinterpret_cast<uint8 *>(1), 0);
	check(CSSM_UnwrapKey(handle(), optionalPublicKey, &keyToBeUnwrapped,
						 spec.usage, spec.attributes, spec.label,
						 &compositeRcc(), &unwrappedKey, &data));
}


Key
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						CssmData *descriptiveData)
{
	Key unwrappedKey;
	check(CSSM_UnwrapKey(handle(), NULL, &keyToBeUnwrapped, spec.usage,
						 spec.attributes, spec.label, &compositeRcc(),
						 unwrappedKey.makeNewKey(attachment()),
						 descriptiveData));
	unwrappedKey->activate();

	return unwrappedKey;
}

void
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						CssmKey &unwrappedKey, CssmData *descriptiveData)
{
	check(CSSM_UnwrapKey(handle(), NULL, &keyToBeUnwrapped, spec.usage,
						 spec.attributes, spec.label, &compositeRcc(),
						 &unwrappedKey, descriptiveData));
}

Key
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						const Key &optionalPublicKey, CssmData *descriptiveData)
{
	Key unwrappedKey;
	check(CSSM_UnwrapKey(handle(), optionalPublicKey, &keyToBeUnwrapped,
						 spec.usage, spec.attributes, spec.label,
						 &compositeRcc(),
						 unwrappedKey.makeNewKey(attachment()),
						 descriptiveData));
	unwrappedKey->activate();

	return unwrappedKey;
}

void
UnwrapKey::operator () (const CssmKey &keyToBeUnwrapped, const KeySpec &spec,
						CssmKey &unwrappedKey, CssmData *descriptiveData,
						const CssmKey *optionalPublicKey)
{
	check(CSSM_UnwrapKey(handle(), optionalPublicKey, &keyToBeUnwrapped,
						 spec.usage, spec.attributes, spec.label,
						 &compositeRcc(), &unwrappedKey,
						 descriptiveData));
}


void DeriveKey::activate()
{
	if (!mActive)
	{
        check(CSSM_CSP_CreateDeriveKeyContext(attachment()->handle(), mAlgorithm,
            mTargetType, mKeySize, mCred, mKey, mIterationCount, mSalt, mSeed, &mHandle));
		mActive = true;
    }
}


Key
DeriveKey::operator () (CssmData *param, const KeySpec &spec)
{
	Key derivedKey;
	check(CSSM_DeriveKey(handle(), param, spec.usage, spec.attributes,
						 spec.label, &compositeRcc(),
						 derivedKey.makeNewKey(attachment())));
	derivedKey->activate();

	return derivedKey;
}

void
DeriveKey::operator () (CssmData *param, const KeySpec &spec,
						CssmKey &derivedKey)
{
	check(CSSM_DeriveKey(handle(), param, spec.usage, spec.attributes,
						 spec.label, &compositeRcc(), &derivedKey));
}

} // end namespace CssmClient
} // end namespace Security
