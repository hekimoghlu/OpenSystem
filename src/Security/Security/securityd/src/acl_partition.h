/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
// acl_partition - "ignore" ACL subject type
//
// A pseudo-ACL that stores partition identifier data.
//
// A PartitionAclSubject always fails to verify.
//
#ifndef _ACL_PARTITION
#define _ACL_PARTITION

#include <security_cdsa_utilities/cssmacl.h>
#include <security_utilities/cfutilities.h>


namespace Security
{

//
// The ANY subject simply matches everything. No sweat.
//
class PartitionAclSubject : public AclSubject {
public:
	PartitionAclSubject()
	: AclSubject(CSSM_ACL_SUBJECT_TYPE_PARTITION), payload(Allocator::standard()) { }
	PartitionAclSubject(Allocator& alloc, const CssmData &data)
	: AclSubject(CSSM_ACL_SUBJECT_TYPE_PARTITION), payload(alloc, data) { }
	
public:
	CssmAutoData payload;
	CFDictionaryRef createDictionaryPayload() const;
	void setDictionaryPayload(Allocator& alloc, CFDictionaryRef dict);
	
public:
	bool validates(const AclValidationContext &ctx) const;
	CssmList toList(Allocator &alloc) const;

    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);


    virtual CFStringRef createACLDebugString() const;

	class Maker : public AclSubject::Maker {
	public:
		Maker() : AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PARTITION) { }
		PartitionAclSubject *make(const TypedList &list) const;
    	PartitionAclSubject *make(Version, Reader &pub, Reader &priv) const;
	};
};

} // end namespace Security


#endif //_ACL_PARTITION
