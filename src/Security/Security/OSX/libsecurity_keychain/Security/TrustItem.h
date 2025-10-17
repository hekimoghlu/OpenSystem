/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
// TrustStore.h - Abstract interface to permanent user trust assignments
//
#ifndef _SECURITY_TRUSTITEM_H_
#define _SECURITY_TRUSTITEM_H_

#include <security_keychain/Item.h>
#include <security_keychain/Certificate.h>
#include <security_keychain/Policies.h>
#include <Security/SecTrustPriv.h>


namespace Security {
namespace KeychainCore {


//
// A trust item in a keychain.
// Currently, Item constructors do not explicitly generate this subclass.
// They don't need to, since our ownly user (TrustStore) can deal with
// the generic Item class just fine.
// If we ever need Item to produce UserTrustItem impls, we would need to
// add constructors from primary key (see Certificate for an example).
//
class UserTrustItem : public ItemImpl {
	NOCOPY(UserTrustItem)
public:	
	struct TrustData {
		Endian<uint32> version;					// version mark
		Endian<SecTrustUserSetting> trust;		// user's trust choice
	};
	static const uint32 currentVersion = 0x101;

public:
	// new item constructor
    UserTrustItem(Certificate *cert, Policy *policy, const TrustData &trust);
    virtual ~UserTrustItem();

	TrustData trust();
	
public:
	static void makeCertIndex(Certificate *cert, CssmOwnedData &index);

protected:
	virtual PrimaryKey add(Keychain &keychain);

	void populateAttributes();

private:
	SecPointer<Certificate> mCertificate;
	SecPointer<Policy> mPolicy;
};


} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_TRUSTITEM_H_
