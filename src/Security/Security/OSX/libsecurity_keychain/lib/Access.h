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
// Access.h - Access control wrappers
//
#ifndef _SECURITY_ACCESS_H_
#define _SECURITY_ACCESS_H_

#include <security_keychain/ACL.h>
#include <security_utilities/trackingallocator.h>
#include <security_cdsa_utilities/cssmaclpod.h>
#include <security_cdsa_utilities/cssmacl.h>
#include <security_cdsa_client/aclclient.h>
#include <security_keychain/TrustedApplication.h>
#include <map>

namespace Security {
namespace KeychainCore {

using CssmClient::AclBearer;


class Access : public SecCFObject {
	NOCOPY(Access)
public:
	SECCFFUNCTIONS(Access, SecAccessRef, errSecInvalidItemRef, gTypes().Access)

	class Maker {
		NOCOPY(Maker)
		static const size_t keySize = 16;	// number of (random) bytes
		friend class Access;
	public:
		enum MakerType {kStandardMakerType, kAnyMakerType};
	
		Maker(Allocator &alloc = Allocator::standard(), MakerType makerType = kStandardMakerType);
		
		void initialOwner(ResourceControlContext &ctx, const AccessCredentials *creds = NULL);
		const AccessCredentials *cred();
		
		TrackingAllocator allocator;
		
		static const char creationEntryTag[];

		MakerType makerType() {return mMakerType;}
		
	private:
		CssmAutoData mKey;
		AclEntryInput mInput;
		AutoCredentials mCreds;
		MakerType mMakerType;
	};

public:
	// make default forms
    Access(const string &description);
    Access(const string &description, const ACL::ApplicationList &trusted);
    Access(const string &description, const ACL::ApplicationList &trusted,
		const AclAuthorizationSet &limitedRights, const AclAuthorizationSet &freeRights);
	
	// make a completely open Access (anyone can do anything)
	Access();
	
	// retrieve from an existing AclBearer
	Access(AclBearer &source);
	
	// make from CSSM layer information (presumably retrieved by caller)
	Access(const CSSM_ACL_OWNER_PROTOTYPE &owner,
		uint32 aclCount, const CSSM_ACL_ENTRY_INFO *acls);
    virtual ~Access();

public:
	CFArrayRef copySecACLs() const;
	CFArrayRef copySecACLs(CSSM_ACL_AUTHORIZATION_TAG action) const;
	
	void add(ACL *newAcl);
	void addOwner(ACL *newOwnerAcl);
	
	void setAccess(AclBearer &target, bool update = false);
	void setAccess(AclBearer &target, Maker &maker);

    void editAccess(AclBearer &target, bool update, const AccessCredentials *cred);

	template <class Container>
	void findAclsForRight(AclAuthorization right, Container &cont)
	{
		cont.clear();
		for (Map::const_iterator it = mAcls.begin(); it != mAcls.end(); it++)
			if (it->second->authorizes(right))
				cont.push_back(it->second);
	}

    // findAclsForRight may return ACLs that have an empty authorization list (and thus "authorize everything")
    // or CSSM_ACL_AUTHORIZATION_ANY, but sometimes you need positive confirmation of a right.
    template <class Container>
    void findSpecificAclsForRight(AclAuthorization right, Container &cont)
    {
        cont.clear();
        for (Map::const_iterator it = mAcls.begin(); it != mAcls.end(); it++)
            if (it->second->authorizesSpecifically(right))
                cont.push_back(it->second);
    }

    // Remove all acl entries that refer to this right.
    void removeAclsForRight(AclAuthorization right);
	
	std::string promptDescription() const;	// from any one of the ACLs contained
	
	void addApplicationToRight(AclAuthorization right, TrustedApplication *app);
	
	void copyOwnerAndAcl(CSSM_ACL_OWNER_PROTOTYPE * &owner,
		uint32 &aclCount, CSSM_ACL_ENTRY_INFO * &acls);
	
protected:
    void makeStandard(const string &description, const ACL::ApplicationList &trusted,
		const AclAuthorizationSet &limitedRights = AclAuthorizationSet(),
		const AclAuthorizationSet &freeRights = AclAuthorizationSet());
    void compile(const CSSM_ACL_OWNER_PROTOTYPE &owner,
        uint32 aclCount, const CSSM_ACL_ENTRY_INFO *acls);


private:
	static const CSSM_ACL_HANDLE ownerHandle = ACL::ownerHandle;
	typedef map<CSSM_ACL_HANDLE, SecPointer<ACL> > Map;

	Map mAcls;			// set of ACL entries
	Mutex mMutex;
};


} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_ACCESS_H_
