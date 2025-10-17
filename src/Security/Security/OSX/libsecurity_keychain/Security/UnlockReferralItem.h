/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
// UnlockReferralItem.h - Abstract interface to permanent user trust assignments
//
#ifndef _SECURITY_UNLOCKREFERRAL_H_
#define _SECURITY_UNLOCKREFERRAL_H_

#include <security_keychain/Item.h>


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
class UnlockReferralItem : public ItemImpl {
	NOCOPY(UnlockReferralItem)
public:	

public:
	// new item constructor
    UnlockReferralItem();
    virtual ~UnlockReferralItem();

protected:
	virtual PrimaryKey add(Keychain &keychain);

	void populateAttributes();

private:
};


} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_UNLOCKREFERRAL_H_
