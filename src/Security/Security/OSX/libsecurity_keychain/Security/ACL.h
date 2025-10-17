/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
// ACL.h - ACL control wrappers
//
#ifndef _SECURITY_ACL_H_
#define _SECURITY_ACL_H_

#include <Security/SecACL.h>
#include <security_cdsa_utilities/cssmaclpod.h>
#include <security_cdsa_client/aclclient.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_utilities/seccfobject.h>
#include "SecCFTypes.h"

#include <vector>

namespace Security {
namespace KeychainCore {

using CssmClient::AclBearer;

class Access;
class TrustedApplication;


//
// An ACL Entry for an Access object
//
class ACL : public SecCFObject {
	NOCOPY(ACL)
public:
	SECCFFUNCTIONS(ACL, SecACLRef, errSecInvalidItemRef, gTypes().ACL)

    // Query AclBearer for ACL entry matching tag. Will throw if there is not exactly 1 entry.
    ACL(const AclBearer &aclBearer, const char *selectionTag,
             Allocator &alloc = Allocator::standard());
	// create from CSSM layer ACL entry
	ACL(const AclEntryInfo &info,
		Allocator &alloc = Allocator::standard());
	// create from CSSM layer owner prototype
	ACL(const AclOwnerPrototype &owner,
		Allocator &alloc = Allocator::standard());
	// create an "any" ACL
	ACL(Allocator &alloc = Allocator::standard());
	// create from "standard form" arguments (with empty application list)
	ACL(string description, const CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR &promptSelector,
		Allocator &alloc = Allocator::standard());
    // create an "integrity" ACL
    ACL(const CssmData &digest, Allocator &alloc = Allocator::standard());
    
    virtual ~ACL();

	Allocator &allocator;
	
	enum State {
		unchanged,					// unchanged from source
		inserted,					// new
		modified,					// was changed (replace)
		deleted						// was deleted (now invalid)
	};
	State state() const { return mState; }
	
	enum Form {
		invalidForm,				// invalid
		customForm,					// not a recognized format (but valid)
		allowAllForm,				// indiscriminate
		appListForm,				// list of apps + prompt confirm
        integrityForm               // hashed integrity of item attributes
	};
	Form form() const { return mForm; }
	void form(Form f) { mForm = f; }
	
    void setIntegrity(const CssmData& integrity);
    const CssmData& integrity();

public:
	AclAuthorizationSet &authorizations()	{ return mAuthorizations; }
	bool authorizes(AclAuthorization right);
    bool authorizesSpecifically(AclAuthorization right);
	void setAuthorization(CSSM_ACL_AUTHORIZATION_TAG auth)
	{ mAuthorizations.clear(); mAuthorizations.insert(auth); }
	
	typedef vector< SecPointer<TrustedApplication> > ApplicationList;
	ApplicationList &applications()
	{ assert(form() == appListForm); return mAppList; }
	void addApplication(TrustedApplication *app);
	
	CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR &promptSelector()	{ return mPromptSelector; }
	string &promptDescription()							{ return mPromptDescription; }
	
	CSSM_ACL_HANDLE entryHandle() const	{ return mCssmHandle; }
	
	static const CSSM_ACL_HANDLE ownerHandle = 0xff0e2743;	// pseudo-handle for owner ACL
	bool isOwner() const			{ return mCssmHandle == ownerHandle; }
	void makeOwner()				{ mCssmHandle = ownerHandle; }
	
	void modify();					// mark modified (update on commit)
	void remove();					// mark removed (delete on commit)
	
	// produce chunk copies of CSSM forms; caller takes ownership
	void copyAclEntry(AclEntryPrototype &proto, Allocator &alloc = Allocator::standard());
	void copyAclOwner(AclOwnerPrototype &proto, Allocator &alloc = Allocator::standard());
	
public:
	void setAccess(AclBearer &target, bool update = false,
		const AccessCredentials *cred = NULL);

public:
	struct ParseError { };
	
public:
	static const CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR defaultSelector;
	
private:
	void parse(const TypedList &subject);
	void parsePrompt(const TypedList &subject);
	void makeSubject();
	void clearSubjects(Form newForm);

private:
	State mState;					// change state
	Form mForm;						// format type

	// AclEntryPrototype fields (minus subject, which is virtually constructed)
	CSSM_ACL_HANDLE mCssmHandle;	// CSSM entry handle (for updates)
	string mEntryTag;				// CSSM entry tag (64 bytes or so, they say)
	bool mDelegate;					// CSSM delegate flag
	AclAuthorizationSet mAuthorizations; // rights for this ACL entry
	
	// composite AclEntryPrototype (constructed when needed)
	TypedList *mSubjectForm;
	
	// following values valid only if form() == appListForm
	ApplicationList mAppList;		// list of trusted applications
    CssmAutoData mIntegrity;          // digest for integrityForm ACL entries
	CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR mPromptSelector; // selector field of PROMPT subject
	string mPromptDescription;		// description field of PROMPT subject
	Mutex mMutex;
};


} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_ACL_H_
