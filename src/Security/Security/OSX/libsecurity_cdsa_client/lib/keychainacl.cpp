/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
// keychainacl - Keychain-related ACL and credential forms
//
#ifdef __MWERKS__
#define _CPP_KEYCHAINACL
#endif

#include "keychainacl.h"
#include <security_cdsa_utilities/cssmwalkers.h>

using namespace CssmClient;


//
// Construct the factory.
// @@@ Leaks.
//
KeychainAclFactory::KeychainAclFactory(Allocator &alloc)
: allocator(alloc), nullCred(alloc, 1), kcCred(alloc, 2), kcUnlockCred(alloc, 1)
{
	// the credential objects self-initialize to empty
	nullCred.sample(0) = TypedList(alloc, CSSM_SAMPLE_TYPE_THRESHOLD);
	
	kcCred.sample(0) = TypedList(alloc, CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT);
	kcCred.sample(1) = TypedList(alloc, CSSM_SAMPLE_TYPE_THRESHOLD,
		new(alloc) ListElement(TypedList(alloc, CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT)));

	// @@@ This leaks a ListElement(CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT)
	kcUnlockCred.sample(0) = TypedList(alloc, CSSM_SAMPLE_TYPE_KEYCHAIN_LOCK,
									  new(alloc) ListElement(CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT));
}

KeychainAclFactory::~KeychainAclFactory()
{
}


//
// Produce credentials.
// These are constants that don't need to be allocated per use.
//
const AccessCredentials *KeychainAclFactory::nullCredentials()
{
	return &nullCred;
}

const AccessCredentials *KeychainAclFactory::keychainPromptCredentials()
{
	return &kcCred;
}

const AccessCredentials *KeychainAclFactory::keychainPromptUnlockCredentials()
{
	return &kcUnlockCred;
}

const AutoCredentials *KeychainAclFactory::passwordChangeCredentials(const CssmData &password)
{
	AutoCredentials *cred = new AutoCredentials(allocator, 1);
	// @@@ This leaks a ListElement(CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT) and ListElement(password)
	cred->sample(0) = TypedList(allocator, CSSM_SAMPLE_TYPE_KEYCHAIN_CHANGE_LOCK,
								new(allocator) ListElement(CSSM_SAMPLE_TYPE_PASSWORD),
								new(allocator) ListElement(password));
	return cred;
}

const AutoCredentials *KeychainAclFactory::passwordUnlockCredentials(const CssmData &password)
{
	AutoCredentials *cred = new AutoCredentials(allocator, 1);
	// @@@ This leaks a ListElement(CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT) and ListElement(password)
	cred->sample(0) = TypedList(allocator, CSSM_SAMPLE_TYPE_KEYCHAIN_LOCK,
								new(allocator) ListElement(CSSM_SAMPLE_TYPE_PASSWORD),
								new(allocator) ListElement(password));
	return cred;
}


//
// 
AclEntryInput *KeychainAclFactory::keychainPromptOwner(const CssmData &description)
{
	// @@@ Make sure this works for a NULL description
	AclEntryPrototype proto(TypedList(allocator, CSSM_ACL_SUBJECT_TYPE_KEYCHAIN_PROMPT,
		new(allocator) ListElement(allocator, description)));
	return new(allocator) AclEntryInput(proto);
}

AclEntryInput *KeychainAclFactory::anyOwner()
{
	AclEntryPrototype proto(TypedList(allocator, CSSM_ACL_SUBJECT_TYPE_ANY));
	return new(allocator) AclEntryInput(proto);
}

void KeychainAclFactory::release(AclEntryInput *input)
{
	DataWalkers::chunkFree(input, allocator);
}
