/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
// defaultcreds - default computations for keychain open credentials
//
#ifndef _SECURITY_DEFAULTCREDS_H
#define _SECURITY_DEFAULTCREDS_H

#include <Security/SecBase.h>
#include <security_cdsa_utilities/cssmcred.h>
#include <security_utilities/trackingallocator.h>
#include <security_cdsa_client/dlclient.h>
#include <security_cdsa_client/dl_standard.h>
#include <vector>
#include <set>


namespace Security {
namespace KeychainCore {


class Keychain;
class KeychainImpl;
class Item;


//
// DefaultCredentials is a self-constructing AccessCredentials variant
// that performs the magic "where are ways to unlock this keychain?" search.
//
class DefaultCredentials : public TrackingAllocator, public AutoCredentials {
public:
	DefaultCredentials(KeychainImpl *kcImpl, Allocator &alloc = Allocator::standard());
	
	bool operator () (CssmClient::Db database);
	
	void clear();
	
private:
	typedef vector<Keychain> KeychainList;

	void keyReferral(const CssmClient::UnlockReferralRecord &ref);
	bool unlockKey(const CssmClient::UnlockReferralRecord &ref, const KeychainList &list);
	
	void keybagReferral(const CssmClient::UnlockReferralRecord &ref);

	KeychainList fallbackSearchList(const DLDbIdentifier &ident);
	
private:
	bool mMade;						// we did it already
	set<Item> mNeededItems;			// Items we need to keep around for unlock use
	KeychainImpl *mKeychainImpl;
};

            
} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_DEFAULTCREDS_H
