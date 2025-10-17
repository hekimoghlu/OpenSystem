/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
// acl_any - "anyone" ACL subject type.
//
// This subject will categorically match everything and anything, even no
// credentials at all (a NULL AccessCredentials pointer).
//
#ifndef _ACL_ANY
#define _ACL_ANY

#include <security_cdsa_utilities/cssmacl.h>
#include <string>

namespace Security {


//
// The ANY subject simply matches everything. No sweat.
//
class AnyAclSubject : public AclSubject {
public:
    AnyAclSubject() : AclSubject(CSSM_ACL_SUBJECT_TYPE_ANY) { }
	bool validates(const AclValidationContext &ctx) const;
	CssmList toList(Allocator &alloc) const;

    virtual CFStringRef createACLDebugString() const;

	class Maker : public AclSubject::Maker {
	public:
		Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_ANY) { }
		AnyAclSubject *make(const TypedList &list) const;
    	AnyAclSubject *make(Version, Reader &pub, Reader &priv) const;
	};
};

} // end namespace Security


#endif //_ACL_ANY
