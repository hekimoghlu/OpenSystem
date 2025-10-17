/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include <Foundation/Foundation.h>
#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecItemPriv.h>
#include <Security/SecInternal.h>
#include <utilities/SecFileLocations.h>
#include <utilities/SecCFWrappers.h>
#include <Security/SecItemBackup.h>

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "secd_regressions.h"

#include "keychain/securityd/SecItemServer.h"

#include "SecdTestKeychainUtilities.h"

void SecAccessGroupsSetCurrent(CFArrayRef accessGroups);
CFArrayRef SecAccessGroupsGetCurrent(void);


static void AddItem(NSDictionary *attr)
{
    NSMutableDictionary *mattr = [attr mutableCopy];
    mattr[(__bridge id)kSecValueData] = [NSData dataWithBytes:"foo" length:3];
    mattr[(__bridge id)kSecAttrAccessible] = (__bridge id)kSecAttrAccessibleAfterFirstUnlock;
    ok_status(SecItemAdd((__bridge CFDictionaryRef)mattr, NULL));
}

int secd_37_pairing_initial_sync(int argc, char *const *argv)
{
    CFErrorRef error = NULL;
    CFTypeRef stuff = NULL;
    OSStatus res = 0;

    plan_tests(16);

    /* custom keychain dir */
    secd_test_setup_temp_keychain("secd_37_pairing_initial_sync", NULL);

    CFArrayRef currentACL = CFRetainSafe(SecAccessGroupsGetCurrent());

    NSMutableArray *newACL = [NSMutableArray arrayWithArray:(__bridge NSArray *)currentACL];
    [newACL addObjectsFromArray:@[
        @"com.apple.ProtectedCloudStorage",
    ]];

    SecAccessGroupsSetCurrent((__bridge CFArrayRef)newACL);


    NSDictionary *pcsinetattrs = @{
        (__bridge id)kSecClass : (__bridge id)kSecClassInternetPassword,
        (__bridge id)kSecAttrAccessGroup : @"com.apple.ProtectedCloudStorage",
        (__bridge id)kSecAttrAccount : @"1",
        (__bridge id)kSecAttrServer : @"current",
        (__bridge id)kSecAttrType : @(0x10001),
        (__bridge id)kSecAttrSynchronizable : @YES,
        (__bridge id)kSecAttrSyncViewHint :  (__bridge id)kSecAttrViewHintPCSMasterKey,
    };
    NSDictionary *pcsinetattrsNotCurrent = @{
        (__bridge id)kSecClass : (__bridge id)kSecClassInternetPassword,
        (__bridge id)kSecAttrAccessGroup : @"com.apple.ProtectedCloudStorage",
        (__bridge id)kSecAttrAccount : @"1",
        (__bridge id)kSecAttrServer : @"noncurrent",
        (__bridge id)kSecAttrType : @(0x00001),
        (__bridge id)kSecAttrSynchronizable : @YES,
        (__bridge id)kSecAttrSyncViewHint :  (__bridge id)kSecAttrViewHintPCSMasterKey,
    };
    NSDictionary *pcsgenpattrs = @{
       (__bridge id)kSecClass : (__bridge id)kSecClassGenericPassword,
       (__bridge id)kSecAttrAccessGroup : @"com.apple.ProtectedCloudStorage",
       (__bridge id)kSecAttrAccount : @"2",
       (__bridge id)kSecAttrSynchronizable : @YES,
       (__bridge id)kSecAttrSyncViewHint :  (__bridge id)kSecAttrViewHintPCSMasterKey,
    };
    NSDictionary *ckksattrs = @{
        (__bridge id)kSecClass : (__bridge id)kSecClassInternetPassword,
        (__bridge id)kSecAttrAccessGroup : @"com.apple.security.ckks",
        (__bridge id)kSecAttrAccount : @"2",
        (__bridge id)kSecAttrSynchronizable : @YES,
        (__bridge id)kSecAttrSyncViewHint :  (__bridge id)kSecAttrViewHintPCSMasterKey,
    };
    AddItem(pcsinetattrs);
    AddItem(pcsinetattrsNotCurrent);
    AddItem(pcsgenpattrs);
    AddItem(ckksattrs);

    uint64_t tlks = 0;
    uint64_t pcs = 0;
    uint64_t bluetooth = 0;

    CFArrayRef items = _SecServerCopyInitialSyncCredentials(SecServerInitialSyncCredentialFlagTLK | SecServerInitialSyncCredentialFlagPCS, &tlks, &pcs, &bluetooth, &error);
    ok(items, "_SecServerCopyInitialSyncCredentials: %@", error);
    CFReleaseNull(error);

    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsinetattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsinetattrsNotCurrent, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsgenpattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)ckksattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);


    ok(_SecItemDeleteAll(&error), "SecItemServerDeleteAll: %@", error);
    CFReleaseNull(error);

    ok(_SecServerImportInitialSyncCredentials(items, &error), "_SecServerImportInitialSyncCredentials: %@", error);
    CFReleaseNull(error);
    CFReleaseNull(items);

    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsinetattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    is_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsinetattrsNotCurrent, &stuff)), errSecItemNotFound,
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)pcsgenpattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);
    ok_status((res = SecItemCopyMatching((__bridge CFDictionaryRef)ckksattrs, &stuff)),
              "SecItemCopyMatching: %d", (int)res);
    CFReleaseNull(stuff);

    SecAccessGroupsSetCurrent(currentACL);
    CFReleaseNull(currentACL);

    secd_test_teardown_delete_temp_keychain("secd_37_pairing_initial_sync");

    return 0;
}
