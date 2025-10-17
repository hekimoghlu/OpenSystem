/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
// DefaultKeychain.h - Private "globals" and interfaces for KeychainCore
//
#ifndef _SECURITY_GLOBALS_H_
#define _SECURITY_GLOBALS_H_

#ifdef check
#undef check
#endif
#include <security_keychain/StorageManager.h>
#include <security_cdsa_client/aclclient.h>


namespace Security
{

namespace KeychainCore
{

class Globals
{
public:
    Globals();
	
	const AccessCredentials *keychainCredentials();
	const AccessCredentials *smartcardCredentials();
	const AccessCredentials *itemCredentials();
	const AccessCredentials *smartcardItemCredentials();

	void setUserInteractionAllowed(bool bUI) { mUI=bUI; }
	bool getUserInteractionAllowed() const { return mUI; }

	// Public globals
	StorageManager storageManager;

    bool integrityProtection() { return mIntegrityProtection; }

private:

	// Other "globals"
	bool mUI;
    bool mIntegrityProtection;
	CssmClient::AclFactory mACLFactory;
};

extern ModuleNexus<Globals> globals;
extern bool gServerMode;

} // end namespace KeychainCore

} // end namespace Security

extern "C" bool GetServerMode();

#endif // !_SECURITY_GLOBALS_H_
