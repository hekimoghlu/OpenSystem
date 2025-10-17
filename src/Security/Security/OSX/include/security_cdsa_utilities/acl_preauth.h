/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
// acl_preauth - a subject type for modeling PINs and similar slot-specific
//		pre-authentication schemes.
//
#ifndef _ACL_PREAUTH
#define _ACL_PREAUTH

#include <security_cdsa_utilities/cssmacl.h>
#include <string>


namespace Security {
namespace PreAuthorizationAcls {


class OriginMaker : public AclSubject::Maker {
protected:
	typedef LowLevelMemoryUtilities::Reader Reader;
	typedef LowLevelMemoryUtilities::Writer Writer;
public:
	OriginMaker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PREAUTH) { }
	AclSubject *make(const TypedList &list) const;
	AclSubject *make(AclSubject::Version version, Reader &pub, Reader &priv) const;
};

class SourceMaker : public AclSubject::Maker {
protected:
	typedef LowLevelMemoryUtilities::Reader Reader;
	typedef LowLevelMemoryUtilities::Writer Writer;
public:
	SourceMaker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PREAUTH_SOURCE) { }
	AclSubject *make(const TypedList &list) const;
	AclSubject *make(AclSubject::Version version, Reader &pub, Reader &priv) const;
};


//
// The actual designation of the PreAuth source AclBearer is provide by the environment.
//
class Environment : public virtual AclValidationEnvironment {
public:
	virtual ObjectAcl *preAuthSource() = 0;
};


//
// This is the object that is being "attached" (as an Adornment) to hold
// the pre-authorization state of a SourceAclSubject.
// The Adornable used for storage is determined by the Environment's store() method.
// 
struct AclState {
	AclState() : accepted(false) { }
	bool accepted;						// was previously accepted by upstream
};


//
// This is the "origin" subject class that gets created the usual way.
// It models a pre-auth "origin" - i.e. it points at a preauth slot and accepts
// its verdict on validation. Think of it as the "come from" part of the link.
//
class OriginAclSubject : public AclSubject {
public:
    bool validates(const AclValidationContext &ctx) const;
    CssmList toList(Allocator &alloc) const;
    
    OriginAclSubject(AclAuthorization auth);
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

private:
	AclAuthorization mAuthTag;		// authorization tag referred to (origin only)
};


//
// The "source" subject class describes the other end of the link; the "go to" part
// if you will. Its sourceSubject is consulted for actual validation; and prior validation
// state is remembered (through the environment store facility) so that future validation
// attempts will automaticaly succeed (that's the "pre" in PreAuth).
//
class SourceAclSubject : public AclSubject {
public:
	bool validates(const AclValidationContext &ctx) const;
	CssmList toList(Allocator &alloc) const;
	
	SourceAclSubject(AclSubject *subSubject,
		CSSM_ACL_PREAUTH_TRACKING_STATE state = CSSM_ACL_PREAUTH_TRACKING_UNKNOWN);
	
	void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
	void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

private:
	RefPointer<AclSubject> mSourceSubject;	// subject determining outcome (source only)
};



}	//  namespace PreAuthorizationAcls
}	//  namespace Security


#endif //_ACL_PREAUTH
