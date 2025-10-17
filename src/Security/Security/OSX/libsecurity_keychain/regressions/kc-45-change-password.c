/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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

int kc_45_change_password(int argc, char *const *argv)
{
    plan_tests(16);

    initializeKeychainTests(__FUNCTION__);

    ok_status(SecKeychainSetUserInteractionAllowed(FALSE), "SecKeychainSetUserInteractionAllowed(FALSE)");

    SecKeychainRef keychain = createNewKeychain("test", "before");
    ok_status(SecKeychainLock(keychain), "SecKeychainLock");
    is_status(SecKeychainChangePassword(keychain, 0, NULL, 5, "after"), errSecInteractionNotAllowed, "Change PW w/ null pw while locked");  // We're not responding to prompt so we can't do stuff while locked
    checkPrompts(1, "Prompt to unlock keychain before password change");
    is_status(SecKeychainChangePassword(keychain, 5, "badpw", 5, "after"), errSecAuthFailed, "Change PW w/ bad pw while locked");
    ok_status(SecKeychainUnlock(keychain, 6, "before", true), "SecKeychainUnlock");
    is_status(SecKeychainChangePassword(keychain, 0, NULL, 5, "after"), errSecAuthFailed, "Change PW w/ null pw while unlocked");
    is_status(SecKeychainChangePassword(keychain, 5, "badpw", 5, "after"), errSecAuthFailed, "Change PW w/ bad pw while unlocked");
    ok_status(SecKeychainChangePassword(keychain, 6, "before", 7, "between"), "Change PW w/ good pw while unlocked");
    ok_status(SecKeychainLock(keychain), "SecKeychainLock");
    ok_status(SecKeychainChangePassword(keychain, 7, "between", 7, "after"), "Change PW w/ good pw while locked");
    checkPrompts(0, "Unexpected keychain access prompt");

    ok_status(SecKeychainDelete(keychain), "%s: SecKeychainDelete", testName);
    CFRelease(keychain);

    deleteTestFiles();
    return 0;
}
