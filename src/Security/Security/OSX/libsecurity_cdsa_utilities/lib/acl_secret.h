/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
// acl_secret - secret-validation password ACLs framework.
//
#ifndef _ACL_SECRET
#define _ACL_SECRET

#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmacl.h>
#include <string>


namespace Security {


//
// SecretAclSubject implements AclSubjects that perform their validation by
// passing their secret through some deterministic validation mechanism.
// As a limiting case, the subject can contain the secret itself and validate
// by comparing for equality.
//
// This is not a fully functional ACL subject. You must subclass it.
//
// There are three elements to consider here:
// (1) How to OBTAIN the secret. This is the job of your subclass; SecretAclSubject
//     is agnostic (and abstract) in this respect.
// (2) How to VALIDATE the secret. This is delegated to an environment method,
//     which gets this very subject passed as an argument for maximum flexibility.
// (3) Whether to use a locally stored copy of the secret for validation (by equality)
//     or hand it off to the environment validator. This is fully implemented here.
// This implementation assumes that the secret, whatever it may be, can be stored
// as a (flat) data blob and can be compared for bit-wise equality. No other
// interpretation is required at this level.
//
class SecretAclSubject : public SimpleAclSubject {
public:
    bool validates(const AclValidationContext &ctx, const TypedList &sample) const;
    
    SecretAclSubject(Allocator &alloc, CSSM_ACL_SUBJECT_TYPE type, const CssmData &secret);
    SecretAclSubject(Allocator &alloc, CSSM_ACL_SUBJECT_TYPE type, CssmManagedData &secret);
	SecretAclSubject(Allocator &alloc, CSSM_ACL_SUBJECT_TYPE type, bool doCache);

	bool haveSecret() const		{ return mSecretValid; }
	bool cacheSecret() const	{ return mCacheSecret; }
	
    void secret(const CssmData &secret) const;
    void secret(CssmManagedData &secret) const;
    
    Allocator &allocator;
	
	IFDUMP(void debugDump() const);
    // Note: SecretAclSubject is not a full class, and so doesn't override createACLDebugString. Subclasses should override that as appropriate.

public:
	class Environment : virtual public AclValidationEnvironment {
	public:
		virtual bool validateSecret(const SecretAclSubject *me,
			const AccessCredentials *secret) = 0;
	};

protected:
	// implement this to get your secret (somehow)
	virtual bool getSecret(const AclValidationContext &context,
		const TypedList &sample, CssmOwnedData &secret) const = 0;
	
	const CssmData &secret() const { assert(mSecretValid); return mSecret; }
    
private:
    mutable CssmAutoData mSecret; // locally known secret
	mutable bool mSecretValid;	// mSecret is valid
	bool mCacheSecret;			// cache secret locally and validate from cache
};

} // end namespace Security


#endif //_ACL_SECRET
