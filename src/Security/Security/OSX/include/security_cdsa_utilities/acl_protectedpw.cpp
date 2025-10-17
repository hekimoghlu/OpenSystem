/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include <security_cdsa_utilities/acl_protectedpw.h>
#include <security_utilities/debugging.h>
#include <algorithm>


//
// Construct a password ACL subject
//
ProtectedPasswordAclSubject::ProtectedPasswordAclSubject(Allocator &alloc, const CssmData &password)
    : SimpleAclSubject(CSSM_ACL_SUBJECT_TYPE_PROTECTED_PASSWORD),
    allocator(alloc), mPassword(alloc, password)
{ }

ProtectedPasswordAclSubject::ProtectedPasswordAclSubject(Allocator &alloc, CssmManagedData &password)
    : SimpleAclSubject(CSSM_ACL_SUBJECT_TYPE_PROTECTED_PASSWORD),
    allocator(alloc), mPassword(alloc, password)
{ }


//
// Validate a credential set against this subject
//
bool ProtectedPasswordAclSubject::validates(const AclValidationContext &context,
    const TypedList &sample) const
{
    if (sample.length() == 1) {
        return true;	//@@@ validate against PP
    } else if (sample.length() == 2 && sample[1].type() == CSSM_LIST_ELEMENT_DATUM) {
        const CssmData &password = sample[1];
        return password == mPassword;
    } else
		CssmError::throwMe(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
}


//
// Make a copy of this subject in CSSM_LIST form
//
CssmList ProtectedPasswordAclSubject::toList(Allocator &alloc) const
{
    // the password itself is private and not exported to CSSM
	return TypedList(alloc, CSSM_ACL_SUBJECT_TYPE_PROTECTED_PASSWORD);
}


//
// Create a ProtectedPasswordAclSubject
//
ProtectedPasswordAclSubject *ProtectedPasswordAclSubject::Maker::make(const TypedList &list) const
{
    CssmAutoData password(Allocator::standard(Allocator::sensitive));
    if (list.length() == 1) {
        char pass[] = "secret";
        CssmData password = CssmData::wrap(pass, 6);	        //@@@ get password from PP
        return new ProtectedPasswordAclSubject(Allocator::standard(Allocator::sensitive), password);
    } else {
        ListElement *password;
        crack(list, 1, &password, CSSM_LIST_ELEMENT_DATUM);
        return new ProtectedPasswordAclSubject(Allocator::standard(Allocator::sensitive), *password);
    }
}

ProtectedPasswordAclSubject *ProtectedPasswordAclSubject::Maker::make(Version,
	Reader &pub, Reader &priv) const
{
    Allocator &alloc = Allocator::standard(Allocator::sensitive);
	const void *data; size_t length; priv.countedData(data, length);
	return new ProtectedPasswordAclSubject(alloc, CssmAutoData(alloc, data, length));
}


//
// Export the subject to a memory blob
//
void ProtectedPasswordAclSubject::exportBlob(Writer::Counter &pub, Writer::Counter &priv)
{
	priv.countedData(mPassword);
}

void ProtectedPasswordAclSubject::exportBlob(Writer &pub, Writer &priv)
{
	priv.countedData(mPassword);
}


#ifdef DEBUGDUMP

void ProtectedPasswordAclSubject::debugDump() const
{
	Debug::dump("Protected Password ");
	Debug::dumpData(mPassword.data(), mPassword.length());
}

#endif //DEBUGDUMP

CFStringRef ProtectedPasswordAclSubject::createACLDebugString() const
{
    // Explicitly do not include any of the password data
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<ProtectedPasswordAclSubject>"));
}
