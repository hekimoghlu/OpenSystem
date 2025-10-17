/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include "Globals.h"
#include "KCExceptions.h"
#include <utilities/SecCFRelease.h>


namespace Security {
namespace KeychainCore {

using namespace CssmClient;

ModuleNexus<Globals> globals;
bool gServerMode;

#pragma mark ÑÑÑÑ Constructor/Destructor ÑÑÑÑ

Globals::Globals() :
mUI(true), mIntegrityProtection(false)
{
    //sudo defaults write /Library/Preferences/com.apple.security KeychainIntegrity -bool YES
    CFTypeRef integrity = (CFNumberRef)CFPreferencesCopyValue(CFSTR("KeychainIntegrity"), CFSTR("com.apple.security"), kCFPreferencesAnyUser, kCFPreferencesCurrentHost);

    if (integrity && CFGetTypeID(integrity) == CFBooleanGetTypeID()) {
        mIntegrityProtection = CFBooleanGetValue((CFBooleanRef)integrity);
    } else {
        // preference not set: defaulting to true
        mIntegrityProtection = true;
    }
    CFReleaseSafe(integrity);
}

const AccessCredentials * Globals::keychainCredentials() 
{
	return (mUI ? mACLFactory.unlockCred() : mACLFactory.cancelCred()); 
}

const AccessCredentials * Globals::smartcardCredentials() 
{
	return (mUI ? mACLFactory.promptedPINCred() : mACLFactory.cancelCred()); 
}

const AccessCredentials * Globals::itemCredentials() 
{
	return (mUI ? mACLFactory.promptCred() : mACLFactory.nullCred()); 
}

const AccessCredentials * Globals::smartcardItemCredentials() 
{
	return (mUI ? mACLFactory.promptedPINItemCred() : mACLFactory.cancelCred()); 
}
	
}	// namespace KeychainCore
}	// namespace Security



extern "C" bool GetServerMode()
{
	return Security::KeychainCore::gServerMode;
}
