/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
// acl_comment - "ignore" ACL subject type
//
// This ACL subject is a historical mistake. It has no use in present applications,
// and remains only to support existing keychains with their already-baked item ACLs.
// Do not use this for new applications of ANY kind.
//
// A CommentAclSubject always fails to verify.
// See further (mis-)usage comments in the .cpp.
//
#ifndef _ACL_COMMENT
#define _ACL_COMMENT

#include <security_cdsa_utilities/cssmacl.h>


namespace Security
{

//
// The ANY subject simply matches everything. No sweat.
//
class CommentAclSubject : public AclSubject {
public:
	CommentAclSubject()
	: AclSubject(CSSM_ACL_SUBJECT_TYPE_COMMENT) { }
	
	bool validates(const AclValidationContext &ctx) const;
	CssmList toList(Allocator &alloc) const;

    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);

	class Maker : public AclSubject::Maker {
	public:
		Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_COMMENT) { }
		CommentAclSubject *make(const TypedList &list) const;
    	CommentAclSubject *make(Version, Reader &pub, Reader &priv) const;
	};
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;
};

} // end namespace Security


#endif //_ACL_COMMENT
