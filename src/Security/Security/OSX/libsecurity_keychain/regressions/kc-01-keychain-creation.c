/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#include <stdlib.h>
#include <Security/SecKeychain.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"

int kc_01_keychain_creation(__unused int argc, __unused char *const *argv)
{
	plan_tests(9);

	ok_status(SecKeychainSetUserInteractionAllowed(FALSE), "disable ui");
    SecKeychainRef keychain = createNewKeychain("test", "test");
	SKIP: {
		skip("can't continue without keychain", 2, ok(keychain, "keychain not NULL"));
		
		is(CFGetRetainCount(keychain), 1, "retaincount of created keychain is 1");
	}

	SecKeychainRef keychain2 = NULL;
	ok_status(SecKeychainOpen("test", &keychain2), "SecKeychainOpen");
	SKIP: {
		skip("can't continue without keychain", 2, ok(keychain, "keychain not NULL"));
		CFIndex retCount = CFGetRetainCount(keychain2);
		is(retCount, 2, "retaincount of created+opened keychain is 2"); // 2, because we opened and created the same keychain.
	}
    
    is(keychain, keychain2, "SecKeychainCreate and SecKeychainOpen returned a different handle for the same keychain");

	ok_status(SecKeychainDelete(keychain), "SecKeychainDelete");

    CFRelease(keychain);
    CFRelease(keychain2);

	return 0;
}

int kc_01_corrupt_keychain(__unused int argc, __unused char *const *argv)
{
    plan_tests(2 + getCorruptTestKeychainTests);

    initializeKeychainTests(__FUNCTION__);

    SecKeychainRef keychain = getCorruptTestKeychain();
    isnt(keychain, NULL, "corrupt keychain created");

    SecKeychainStatus kcStatus;
    is(SecKeychainGetStatus(keychain, &kcStatus), errSecInvalidKeychain, "SecKeychainGetStatus returns error");

    CFRelease(keychain);

    return 0;
}
