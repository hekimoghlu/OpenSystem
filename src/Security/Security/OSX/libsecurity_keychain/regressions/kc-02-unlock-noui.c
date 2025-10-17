/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#include <Security/SecKeychain.h>
#include <Security/SecKeychainPriv.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"

int kc_02_unlock_noui(int argc, char *const *argv)
{
    plan_tests(12);

    initializeKeychainTests(__FUNCTION__);

	ok_status(SecKeychainSetUserInteractionAllowed(FALSE), "SecKeychainSetUserInteractionAllowed(FALSE)");

    SecKeychainRef keychain = createNewKeychain("test", "test");
	ok_status(SecKeychainLock(keychain), "SecKeychainLock");

    is_status(SecKeychainUnlock(keychain, 0, NULL, FALSE), errSecAuthFailed, "SecKeychainUnlock");

    checkPrompts(0, "Unexpected keychain access prompt unlocking after SecKeychainCreate");

	ok_status(SecKeychainLock(keychain), "SecKeychainLock");
	CFRelease(keychain);

	ok_status(SecKeychainOpen("test", &keychain), "SecKeychainOpen locked kc");

    is_status(SecKeychainUnlock(keychain, 0, NULL, FALSE), errSecAuthFailed, "SecKeychainUnlock");

    checkPrompts(0, "Unexpected keychain access prompt unlocking after SecKeychainCreate");

    ok_status(SecKeychainDelete(keychain), "%s: SecKeychainDelete", testName);
    CFRelease(keychain);

    deleteTestFiles();
    return 0;
}
