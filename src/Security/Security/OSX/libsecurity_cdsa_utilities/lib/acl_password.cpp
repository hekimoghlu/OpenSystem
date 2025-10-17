/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
// acl_password - password-based ACL subject types
//
#include <security_cdsa_utilities/acl_password.h>
#include <security_utilities/debugging.h>
#include <security_utilities/endian.h>
#include <algorithm>


//
// PasswordAclSubject always pre-loads its secret, and thus never has to
// "get" its secret. If we ever try, it's a bug.
//
bool PasswordAclSubject::getSecret(const AclValidationContext &context,
	const TypedList &sample, CssmOwnedData &secret) const
{
	switch (sample.length()) {
	case 1:
		return false;	// no password in sample
	case 2:
		secret = sample[1];
		return true;
	default:
		CssmError::throwMe(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
	}
}


//
// Make a copy of this subject in CSSM_LIST form
//
CssmList PasswordAclSubject::toList(Allocator &alloc) const
{
    // the password itself is private and not exported to CSSM
	return TypedList(alloc, CSSM_ACL_SUBJECT_TYPE_PASSWORD);
}


//
// Create a PasswordAclSubject
//
PasswordAclSubject *PasswordAclSubject::Maker::make(const TypedList &list) const
{
    Allocator &alloc = Allocator::standard(Allocator::sensitive);
	switch (list.length()) {
	case 1:
		return new PasswordAclSubject(alloc, true);
	case 2:
		{
			ListElement *password;
			crack(list, 1, &password, CSSM_LIST_ELEMENT_DATUM);
			return new PasswordAclSubject(alloc, password->data());
		}
	default:
		CssmError::throwMe(CSSM_ERRCODE_INVALID_ACL_SUBJECT_VALUE);
	}
}

PasswordAclSubject *PasswordAclSubject::Maker::make(Version, Reader &pub, Reader &priv) const
{
    Allocator &alloc = Allocator::standard(Allocator::sensitive);
	const void *data; size_t length; priv.countedData(data, length);
	CssmAutoData passwordData(alloc, data, length);
	return new PasswordAclSubject(alloc, passwordData);
}


//
// Export the subject to a memory blob
//
void PasswordAclSubject::exportBlob(Writer::Counter &pub, Writer::Counter &priv)
{
	priv.countedData(secret());
}

void PasswordAclSubject::exportBlob(Writer &pub, Writer &priv)
{
	priv.countedData(secret());
}


#ifdef DEBUGDUMP

void PasswordAclSubject::debugDump() const
{
	Debug::dump("Password");
	SecretAclSubject::debugDump();
}

#endif //DEBUGDUMP

CFStringRef PasswordAclSubject::createACLDebugString() const
{
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<PasswordAclSubject(type:%d)>"), this->type());
}
