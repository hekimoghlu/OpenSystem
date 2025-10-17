/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include <CoreFoundation/CoreFoundation.h>
#include <TargetConditionals.h>
#include <stdio.h>

#include "keychain_regressions.h"
#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecItemPriv.h>
#include <utilities/SecCFRelease.h>
#include <AppleKeyStore/libaks.h>
#include <AssertMacros.h>
#include "keychain/securityd/SOSCloudCircleServer.h"

#define DATA_ARG(x) (x) ? CFDataGetBytePtr((x)) : NULL, (x) ? (int)CFDataGetLength((x)) : 0

static CFDataRef create_keybag(keybag_handle_t bag_type, CFDataRef password)
{
    keybag_handle_t handle = bad_keybag_handle;
    
    if (aks_create_bag(DATA_ARG(password), bag_type, &handle) == 0) {
        void * keybag = NULL;
        int keybag_size = 0;
        if (aks_save_bag(handle, &keybag, &keybag_size) == 0) {
            return CFDataCreate(kCFAllocatorDefault, keybag, keybag_size);
        }
    }
    
    return CFDataCreate(kCFAllocatorDefault, NULL, 0);
}

/* Test low level keychain migration from device to device interface. */
static void tests(void)
{
    int v_eighty = 80;
    CFNumberRef eighty = CFNumberCreate(NULL, kCFNumberSInt32Type, &v_eighty);
    const char *v_data = "test";
    CFDataRef pwdata = CFDataCreate(NULL, (UInt8 *)v_data, strlen(v_data));
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFTypeRef result = NULL;
    CFDictionaryAddValue(query, kSecClass, kSecClassInternetPassword);
    CFDictionaryAddValue(query, kSecAttrServer, CFSTR("members.spamcop.net"));
    CFDictionaryAddValue(query, kSecAttrAccount, CFSTR("smith"));
    CFDictionaryAddValue(query, kSecAttrPort, eighty); CFReleaseNull(eighty);
    CFDictionaryAddValue(query, kSecAttrProtocol, kSecAttrProtocolHTTP);
    CFDictionaryAddValue(query, kSecAttrAuthenticationType, kSecAttrAuthenticationTypeDefault);
    CFDictionaryAddValue(query, kSecValueData, pwdata); CFReleaseNull(pwdata);
    CFDictionaryAddValue(query, kSecAttrSynchronizable, kCFBooleanTrue);
    
    CFDataRef keybag = NULL;
    const char *p = "sup3rsekretpassc0de";
    CFDataRef password = CFDataCreate(NULL, (UInt8 *)p, strlen(p));
    
    keybag = create_keybag(kAppleKeyStoreAsymmetricBackupBag, password);
    
    SecItemDelete(query);
    
    // add syncable item
    ok_status(SecItemAdd(query, NULL), "add internet password");
    
    ok_status(SecItemCopyMatching(query, &result), "find item we are about to destroy");
    if (result) { CFRelease(result); result = NULL; }

    CFDictionaryRef backup = NULL;
    
    ok_status(_SecKeychainBackupSyncable(keybag, password, NULL, &backup), "export items");
    
    ok_status(SecItemDelete(query), "delete item we backed up");
    is_status(SecItemCopyMatching(query, &result), errSecItemNotFound, "find item we are about to destroy");
    if (result) { CFRelease(result); result = NULL; }
    
    ok_status(_SecKeychainRestoreSyncable(keybag, password, backup), "import items");
    
    ok_status(SecItemCopyMatching(query, &result), "find restored item");
    if (result) { CFRelease(result); result = NULL; }
    
    ok_status(SecItemDelete(query), "delete restored item");
    
    CFReleaseNull(backup);
    CFReleaseNull(keybag);
    CFReleaseNull(query);
    CFReleaseNull(password);
}

int si_33_keychain_backup(int argc, char *const *argv)
{
	plan_tests(8);
    
    CFErrorRef localError = NULL;

    SOSCCSetCompatibilityMode(true, &localError);
    CFReleaseNull(localError);
	
    tests();
    
    SOSCCSetCompatibilityMode(false, &localError);
    CFReleaseNull(localError);

	return 0;
}
