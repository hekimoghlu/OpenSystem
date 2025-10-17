/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
// acl_protectedpw - protected-path password-based ACL subject types.
//
// This implements "protected path" password-based subject types as per CSSM standard.
// A "protected path" is something that is outside the scope of the computer proper,
// like e.g. a PINpad directly attached to a smartcard token.
// Note: A password prompted through securityd/SecurityAgent is a "prompted password",
// not a "protected password". See acl_prompted.h.
//
// @@@ Warning: This is not quite implemented.
//
#ifndef _ACL_PROTECTED_PASSWORD
#define _ACL_PROTECTED_PASSWORD

#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmacl.h>
#include <string>


namespace Security {

class ProtectedPasswordAclSubject : public SimpleAclSubject {
public:
    bool validates(const AclValidationContext &baseCtx, const TypedList &sample) const;
    CssmList toList(Allocator &alloc) const;
    
    ProtectedPasswordAclSubject(Allocator &alloc, const CssmData &password);
    ProtectedPasswordAclSubject(Allocator &alloc, CssmManagedData &password);
    
    Allocator &allocator;
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

    class Maker : public AclSubject::Maker {
    public:
    	Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PROTECTED_PASSWORD) { }
    	ProtectedPasswordAclSubject *make(const TypedList &list) const;
    	ProtectedPasswordAclSubject *make(Version, Reader &pub, Reader &priv) const;
    };
    
private:
    CssmAutoData mPassword;
};

} // end namespace Security


#endif //_ACL_PROTECTED_PASSWORD
