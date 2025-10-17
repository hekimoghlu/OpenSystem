/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#ifndef _KEYCHAINACL
#define _KEYCHAINACL

#include <Security/cssm.h>
#include <security_cdsa_utilities/cssmaclpod.h>
#include <security_cdsa_utilities/cssmcred.h>
#include <security_cdsa_utilities/cssmalloc.h>

#ifdef _CPP_KEYCHAINACL
# pragma export on
#endif


namespace Security
{

namespace CssmClient
{

class KeychainAclFactory
{
public:
	KeychainAclFactory(Allocator &alloc);
	~KeychainAclFactory();
	
	Allocator &allocator;
	
public:
	//
	// Create credentials. These functions return AccessCredentials pointers.
	//
	const AccessCredentials *nullCredentials();
	const AccessCredentials *keychainPromptCredentials();
	const AccessCredentials *keychainPromptUnlockCredentials();
	const AutoCredentials *passwordChangeCredentials(const CssmData &password);
	const AutoCredentials *passwordUnlockCredentials(const CssmData &password);

public:
	//
	// Create initial ACLs. Pass those to resource creation functions.
	//
	AclEntryInput *keychainPromptOwner(const CssmData &description);
	AclEntryInput *anyOwner();
	void release(AclEntryInput *input);
	
private:
	AutoCredentials nullCred;
	AutoCredentials kcCred;
	AutoCredentials kcUnlockCred;
};


} // end namespace CssmClient

} // end namespace Security

#ifdef _CPP_KEYCHAINACL
# pragma export off
#endif

#endif //_KEYCHAINACL
