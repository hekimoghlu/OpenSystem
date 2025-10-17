/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
// acl_password - password-based ACL subject types.
//
// This implements simple password-based subject types as per CSSM standard.
//
#ifndef _ACL_PASSWORD
#define _ACL_PASSWORD

#include <security_cdsa_utilities/acl_secret.h>


namespace Security {


//
// A PasswordAclSubject simply contains its secret.
// The environment is never consulted; we just compare against our known secret.
//
class PasswordAclSubject : public SecretAclSubject {
public:
    CssmList toList(Allocator &alloc) const;
    
    PasswordAclSubject(Allocator &alloc, const CssmData &password)
		: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PASSWORD, password) { }
    PasswordAclSubject(Allocator &alloc, CssmManagedData &password)
		: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PASSWORD, password) { }
	PasswordAclSubject(Allocator &alloc, bool cache)
		: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PASSWORD, cache) { }
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

public:
    class Maker : public SecretAclSubject::Maker {
    public:
    	Maker() : SecretAclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PASSWORD) { }
    	PasswordAclSubject *make(const TypedList &list) const;
    	PasswordAclSubject *make(Version, Reader &pub, Reader &priv) const;
    };
	
protected:
	bool getSecret(const AclValidationContext &context,
		const TypedList &subject, CssmOwnedData &secret) const;
};

} // end namespace Security


#endif //_ACL_PASSWORD
