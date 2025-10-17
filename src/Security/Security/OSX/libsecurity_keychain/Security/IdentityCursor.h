/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
// IdentityCursor.h - Working with IdentityCursors
//
#ifndef _SECURITY_IDENTITYCURSOR_H_
#define _SECURITY_IDENTITYCURSOR_H_

#include <Security/SecCertificate.h>
#include <Security/SecIdentity.h>
#include <Security/SecIdentitySearch.h>
#include <security_cdsa_client/securestorage.h>
#include <security_keychain/KCCursor.h>
#include <CoreFoundation/CFArray.h>

namespace Security
{

namespace KeychainCore
{

class Identity;
class KeyItem;

class IdentityCursor : public SecCFObject
{
    NOCOPY(IdentityCursor)
public:
	SECCFFUNCTIONS(IdentityCursor, SecIdentitySearchRef, errSecInvalidSearchRef, gTypes().IdentityCursor)

    IdentityCursor(const StorageManager::KeychainList &searchList, CSSM_KEYUSE keyUsage);
	virtual ~IdentityCursor() _NOEXCEPT;
	virtual bool next(SecPointer<Identity> &identity);

	CFDataRef pubKeyHashForSystemIdentity(CFStringRef domain);

protected:
	StorageManager::KeychainList mSearchList;

private:
	KCCursor mKeyCursor;
	KCCursor mCertificateCursor;
	SecPointer<KeyItem> mCurrentKey;
	Mutex mMutex;
};

class IdentityCursorPolicyAndID : public IdentityCursor
{
public:
    IdentityCursorPolicyAndID(const StorageManager::KeychainList &searchList, CSSM_KEYUSE keyUsage, CFStringRef idString, SecPolicyRef policy, bool returnOnlyValidIdentities);
	virtual ~IdentityCursorPolicyAndID() _NOEXCEPT;
	virtual bool next(SecPointer<Identity> &identity);
	virtual void findPreferredIdentity();

private:
	SecPolicyRef mPolicy;
	CFStringRef mIDString;
	bool mReturnOnlyValidIdentities;
	bool mPreferredIdentityChecked;
	SecPointer<Identity> mPreferredIdentity;
};


} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_IDENTITYCURSOR_H_
