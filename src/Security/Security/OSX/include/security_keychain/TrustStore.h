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
// TrustStore.h - Abstract interface to permanent user trust assignments
//
#ifndef _SECURITY_TRUSTSTORE_H_
#define _SECURITY_TRUSTSTORE_H_

#include <security_keychain/Certificate.h>
#include <security_keychain/Policies.h>
#include <Security/SecTrust.h>
#include <security_keychain/TrustItem.h>


namespace Security {
namespace KeychainCore {


//
// A TrustStore object mediates access to "user trust" information stored
// for a user (usually in her keychains).
// For lack of a better home, access to the default anchor certificate
// list is also provided here.
//
class TrustStore {
	NOCOPY(TrustStore)
public:
    TrustStore(Allocator &alloc = Allocator::standard());
    virtual ~TrustStore();
	
	Allocator &allocator;

	// set/get user trust for a certificate and policy
    SecTrustUserSetting find(Certificate *cert, Policy *policy, 
		StorageManager::KeychainList &keychainList);
    void assign(Certificate *cert, Policy *policy, SecTrustUserSetting assignment);
    
	void getCssmRootCertificates(CertGroup &roots);
	
	typedef UserTrustItem::TrustData TrustData;
	
protected:
	Item findItem(Certificate *cert, Policy *policy, 
		StorageManager::KeychainList &keychainList);
	void loadRootCertificates();

private:
	bool mRootsValid;			// roots have been loaded from disk
	vector<CssmData> mRoots;	// array of CssmDatas to certificate datas
	CssmAutoData mRootBytes;	// certificate data blobs (bunched up)
    CFRef<CFArrayRef> mCFRoots;	// mRoots as CFArray<SecCertificate>
	Mutex mMutex;
};

} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_TRUSTSTORE_H_
