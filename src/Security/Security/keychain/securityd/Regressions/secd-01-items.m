/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include "secd_regressions.h"

#include "keychain/securityd/SecDbItem.h"
#include "keychain/securityd/SecItemServer.h"

#include <utilities/array_size.h>
#include <utilities/SecFileLocations.h>

#include <unistd.h>

#include "SecdTestKeychainUtilities.h"

#if USE_KEYSTORE
#include "OSX/utilities/SecAKSWrappers.h"

int secd_01_items(int argc, char *const *argv)
{
    plan_tests(27 + kSecdTestSetupTestCount);

    secd_test_setup_testviews(); // if running all tests get the test views setup first
    /* custom keychain dir */
    secd_test_setup_temp_keychain("secd_01_items", NULL);

    /* custom keybag */
    keybag_handle_t keybag;
    keybag_state_t state;
    char *passcode="password";
    int passcode_len=(int)strlen(passcode);

    ok(kAKSReturnSuccess==aks_create_bag(passcode, passcode_len, kAppleKeyStoreDeviceBag, &keybag), "create keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(!(state&keybag_state_locked), "keybag unlocked");
    SecItemServerSetKeychainKeybag(keybag);

    /* lock */
    ok(kAKSReturnSuccess==aks_lock_bag(keybag), "lock keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(state&keybag_state_locked, "keybag locked");

    
    SecKeychainDbReset(NULL);

    /* Creating a password */
    int v_eighty = 80;
    CFNumberRef eighty = CFNumberCreate(NULL, kCFNumberSInt32Type, &v_eighty);
    const char *v_data = "test";
    CFDataRef pwdata = CFDataCreate(NULL, (UInt8 *)v_data, strlen(v_data));
    const void *keys[] = {
        kSecClass,
        kSecAttrServer,
        kSecAttrAccount,
        kSecAttrPort,
        kSecAttrProtocol,
        kSecAttrAuthenticationType,
        kSecReturnData,
        kSecValueData
    };
    const void *values[] = {
        kSecClassInternetPassword,
        CFSTR("members.spamcop.net"),
        CFSTR("smith"),
        eighty,
        CFSTR("http"),
        CFSTR("dflt"),
        kCFBooleanTrue,
        pwdata
    };
    
    CFDictionaryRef item = CFDictionaryCreate(NULL, keys, values,
                                              array_size(keys), NULL, NULL);

    
    is_status(SecItemAdd(item, NULL), errSecInteractionNotAllowed, "add internet password while locked");

    /* unlock */
    ok(kAKSReturnSuccess==aks_unlock_bag(keybag, passcode, passcode_len), "unlock keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(!(state&keybag_state_locked), "keybag unlocked");

    ok_status(SecItemAdd(item, NULL), "add internet password, while unlocked");

    
    /* lock */
    ok(kAKSReturnSuccess==aks_lock_bag(keybag), "lock keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(state&keybag_state_locked, "keybag locked");

    is_status(SecItemAdd(item, NULL), errSecInteractionNotAllowed,
              "add internet password again, while locked");

    /* unlock */
    ok(kAKSReturnSuccess==aks_unlock_bag(keybag, passcode, passcode_len), "unlock keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(!(state&keybag_state_locked), "keybag unlocked");

    is_status(SecItemAdd(item, NULL), errSecDuplicateItem,
              "add internet password again, while unlocked");

    CFTypeRef results = NULL;
    /* Create a dict with all attrs except the data. */
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values,
                                               (array_size(keys)) - 1, NULL, NULL);
    ok_status(SecItemCopyMatching(query, &results), "find internet password, while unlocked ");
    if (results) {
        CFRelease(results);
        results = NULL;
    }

    /* lock */
    ok(kAKSReturnSuccess==aks_lock_bag(keybag), "lock keybag");
    ok(kAKSReturnSuccess==aks_get_lock_state(keybag, &state), "get keybag state");
    ok(state&keybag_state_locked, "keybag locked");

    is_status(SecItemCopyMatching(query, &results), errSecInteractionNotAllowed, "find internet password, while locked ");

    /* Reset keybag and custom $HOME */
    SecItemServerSetKeychainKeybagToDefault();
    secd_test_teardown_delete_temp_keychain("secd_01_items");
    void* buf = NULL;
    int bufLen = 0;
    ok(kAKSReturnSuccess == aks_save_bag(keybag, &buf, &bufLen), "failed to save keybag for invalidation");
    ok(kAKSReturnSuccess == aks_unload_bag(keybag), "failed to unload keybag for invalidation");
    ok(kAKSReturnSuccess == aks_invalidate_bag(buf, bufLen), "failed to invalidate keybag");
    free(buf);

    CFReleaseNull(pwdata);
    CFReleaseNull(eighty);
    CFReleaseSafe(item);
    CFReleaseSafe(query);


	return 0;
}

#else

int secd_01_items(int argc, char *const *argv)
{
    plan_tests(1);

    todo("Not yet working in simulator");

TODO: {
    ok(false);
}

    /* not implemented in simulator (no keybag) */
	return 0;
}
#endif

