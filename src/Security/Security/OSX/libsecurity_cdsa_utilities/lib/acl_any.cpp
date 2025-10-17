/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include <security_cdsa_utilities/acl_any.h>
#include <algorithm>


//
// The ANY subject matches all credentials, including none at all.
//
bool AnyAclSubject::validates(const AclValidationContext &) const
{
	return true;
}

CFStringRef AnyAclSubject::createACLDebugString() const
{
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("AnyAclSubject"));
}

//
// The CSSM_LIST version is trivial. It has no private part to omit.
//
CssmList AnyAclSubject::toList(Allocator &alloc) const
{
	return TypedList(alloc, CSSM_ACL_SUBJECT_TYPE_ANY);
}


//
// The subject form takes no arguments.
//
AnyAclSubject *AnyAclSubject::Maker::make(const TypedList &list) const
{
	crack(list, 0);	// no arguments in input list
	return new AnyAclSubject();
}

AnyAclSubject *AnyAclSubject::Maker::make(Version, Reader &, Reader &) const
{
    return new AnyAclSubject();
}

