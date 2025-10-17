/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

#import <Foundation/Foundation.h>
#include <Security/Security.h>

#include <utilities/SecCFWrappers.h>
#include "SecDbKeychainItem.h"

#include <TargetConditionals.h>

#if USE_KEYSTORE
#include "OSX/utilities/SecAKSWrappers.h"

#include "SecdTestKeychainUtilities.h"

int secd_36_ks_encrypt(int argc, char *const *argv)
{
    plan_tests(11 + kSecdTestSetupTestCount);

    secd_test_setup_temp_keychain("secd_36_ks_encrypt", NULL);

    keybag_handle_t keybag;
    keybag_state_t state;
    CFDictionaryRef data = NULL;
    CFDataRef enc = NULL;
    CFErrorRef error = NULL;
    SecAccessControlRef ac = NULL;
    bool ret;

    char passcode[] = "password";
    int passcode_len = sizeof(passcode) - 1;


    /* Create and lock custom keybag */
    is(kAKSReturnSuccess, aks_create_bag(passcode, passcode_len, kAppleKeyStoreDeviceBag, &keybag), "create keybag");
    is(kAKSReturnSuccess, aks_get_lock_state(keybag, &state), "get keybag state");
    is(0, (int)(state&keybag_state_locked), "keybag unlocked");

    data = (__bridge CFDictionaryRef)@{
        (id)kSecValueData : @"secret here",
    };

    ok(ac = SecAccessControlCreate(NULL, &error), "SecAccessControlCreate: %@", error);
    ok(SecAccessControlSetProtection(ac, kSecAttrAccessibleWhenUnlocked, &error), "SecAccessControlSetProtection: %@", error);

    CFDictionaryRef empty = (__bridge CFDictionaryRef)@{};
    ret = ks_encrypt_data(keybag, ac, NULL, data, (__bridge CFDictionaryRef)@{@"persistref" : @"aaa-bbb-ccc"}, empty, &enc, true, false, &error);
    is(true, ret);

    CFReleaseNull(ac);

    {
        CFMutableDictionaryRef attributes = NULL;
        uint32_t version = 0;

        NSData* dummyACM = [NSData dataWithBytes:"dummy" length:5];
        const SecDbClass* class = kc_class_with_name(kSecClassGenericPassword);
        NSArray* dummyArray = [NSArray array];

        ret = ks_decrypt_data(keybag, NULL, kAKSKeyOpDecrypt, &ac, (__bridge CFDataRef _Nonnull)dummyACM, enc, class, (__bridge CFArrayRef)dummyArray, &attributes, &version, true, NULL, &error);
        is(true, ret, "ks_decrypt_data: %@", error);

        CFTypeRef aclProtection = ac ? SecAccessControlGetProtection(ac) : NULL;
        ok(aclProtection && CFEqual(aclProtection, kSecAttrAccessibleWhenUnlocked), "AccessControl protection is: %@", aclProtection);

        CFReleaseNull(ac);
    }

    CFReleaseNull(error);
    CFReleaseNull(enc);

    secd_test_teardown_delete_temp_keychain("secd_36_ks_encrypt");

    void* buf = NULL;
    int bufLen = 0;
    ok(kAKSReturnSuccess == aks_save_bag(keybag, &buf, &bufLen), "failed to save keybag for invalidation");
    ok(kAKSReturnSuccess == aks_unload_bag(keybag), "failed to unload keybag for invalidation");
    ok(kAKSReturnSuccess == aks_invalidate_bag(buf, bufLen), "failed to invalidate keybag");
    free(buf);

    return 0;
}

#else /* !USE_KEYSTORE */

int secd_36_ks_encrypt(int argc, char *const *argv)
{
    plan_tests(1);
    ok(true);
    return 0;
}
#endif /* USE_KEYSTORE */
