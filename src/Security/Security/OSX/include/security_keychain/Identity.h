/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
// Identity.h - Working with Identities
//
#ifndef _SECURITY_IDENTITY_H_
#define _SECURITY_IDENTITY_H_

#include <security_keychain/Certificate.h>
#include <security_keychain/KeyItem.h>

namespace Security
{

namespace KeychainCore
{

class Identity : public SecCFObject
{
    NOCOPY(Identity)
public:
	SECCFFUNCTIONS(Identity, SecIdentityRef, errSecInvalidItemRef, gTypes().Identity)

    Identity(const SecPointer<KeyItem> &privateKey,
             const SecPointer<Certificate> &certificate);
    Identity(const SecKeyRef privateKey,
             const SecPointer<Certificate> &certificate);
    Identity(const StorageManager::KeychainList &keychains, const SecPointer<Certificate> &certificate);
    virtual ~Identity() _NOEXCEPT;

	SecPointer<KeyItem> privateKey() const;
	SecPointer<Certificate> certificate() const;
    SecKeyRef privateKeyRef() const;

	bool operator < (const Identity &other) const;
	bool operator == (const Identity &other) const;

	bool equal(SecCFObject &other);
    CFHashCode hash();

private:
    SecKeyRef mPrivateKey;
	SecPointer<Certificate> mCertificate;
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_IDENTITY_H_
