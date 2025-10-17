/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
// acl_prompted - password-based validation with out-of-band prompting.
//
#include <security_cdsa_utilities/acl_prompted.h>
#include <security_utilities/debugging.h>
#include <security_utilities/endian.h>
#include <algorithm>


//
// Construct PromptedAclSubjects from prompts and optional data
//
PromptedAclSubject::PromptedAclSubject(Allocator &alloc,
	const CssmData &prompt, const CssmData &password)
	: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PROMPTED_PASSWORD, password),
	  mPrompt(alloc, prompt) { }
PromptedAclSubject::PromptedAclSubject(Allocator &alloc,
	CssmManagedData &prompt, CssmManagedData &password)
	: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PROMPTED_PASSWORD, password),
	  mPrompt(alloc, prompt) { }
PromptedAclSubject::PromptedAclSubject(Allocator &alloc,
	const CssmData &prompt, bool cache)
	: SecretAclSubject(alloc, CSSM_ACL_SUBJECT_TYPE_PROMPTED_PASSWORD, cache),
	  mPrompt(alloc, prompt) { }


//
// PromptedAclSubject will prompt for the secret
//
bool PromptedAclSubject::getSecret(const AclValidationContext &context,
	const TypedList &subject, CssmOwnedData &secret) const
{
	if (Environment *env = context.environment<Environment>()) {
		return env->getSecret(secret, mPrompt);
	} else {
		return false;
	}
}


//
// Make a copy of this subject in CSSM_LIST form
//
CssmList PromptedAclSubject::toList(Allocator &alloc) const
{
    // the password itself is private and not exported to CSSM
	return TypedList(alloc, CSSM_ACL_SUBJECT_TYPE_PROMPTED_PASSWORD,
		new(alloc) ListElement(alloc, mPrompt));
}


//
// Create a PromptedAclSubject
//
PromptedAclSubject *PromptedAclSubject::Maker::make(const TypedList &list) const
{
    Allocator &alloc = Allocator::standard(Allocator::sensitive);
	switch (list.length()) {
	case 2:
		{
			ListElement *elem[1];
			crack(list, 1, elem, CSSM_LIST_ELEMENT_DATUM);
			return new PromptedAclSubject(alloc, elem[0]->data(), true);
		}
	case 3:
		{
			ListElement *elem[2];
			crack(list, 2, elem, CSSM_LIST_ELEMENT_DATUM, CSSM_LIST_ELEMENT_DATUM);
			return new PromptedAclSubject(alloc, elem[0]->data(), elem[1]->data());
		}
	default:
		CssmError::throwMe(CSSM_ERRCODE_INVALID_ACL_SUBJECT_VALUE);
	}
}

PromptedAclSubject *PromptedAclSubject::Maker::make(Version, Reader &pub, Reader &priv) const
{
    Allocator &alloc = Allocator::standard(Allocator::sensitive);
    const void *data; size_t length; priv.countedData(data, length);
	return new PromptedAclSubject(alloc, CssmAutoData(alloc, data, length), true);
}


//
// Export the subject to a memory blob
//
void PromptedAclSubject::exportBlob(Writer::Counter &pub, Writer::Counter &priv)
{
	pub.countedData(mPrompt);
}

void PromptedAclSubject::exportBlob(Writer &pub, Writer &priv)
{
	pub.countedData(mPrompt);
}


#ifdef DEBUGDUMP

void PromptedAclSubject::debugDump() const
{
	Debug::dump("Prompted-Password");
	SecretAclSubject::debugDump();
}

#endif //DEBUGDUMP

CFStringRef PromptedAclSubject::createACLDebugString() const
{
    return CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("<PromptedAclSubject>"));
}
