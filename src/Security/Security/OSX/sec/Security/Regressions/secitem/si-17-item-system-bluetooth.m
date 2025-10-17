/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <Security/SecItemPriv.h>
#include <utilities/array_size.h>
#include <utilities/SecCFRelease.h>
#include <stdlib.h>
#include <unistd.h>

#include "Security_regressions.h"

// If we're running no server this test isn't valid
#if defined(NO_SERVER) && NO_SERVER
#define DO_TESTS false
#else
#define DO_TESTS true
#endif


static NSString* const kBlueToothServiceName = @"BluetoothGlobal";

/* Test presence of properly migrated items into system keychain. */
static void tests(void)
{
    NSDictionary *query;
    NSDictionary *whoami = NULL;

    whoami = CFBridgingRelease(_SecSecuritydCopyWhoAmI(NULL));

    NSLog(@"whoami: %@", whoami);

    /*
     * Check we can't find it in our keychain
     */
    SKIP: {
        skip("No Server mode, test not valid", 2, DO_TESTS)
        query = @{
                  (__bridge id)kSecClass : (__bridge id)kSecClassGenericPassword,
                  (__bridge id)kSecAttrService : kBlueToothServiceName,
                  };

        /*
         * Check for multi user mode and its expected behavior (since its different)
         */

        bool multiUser = (whoami[@"musr"]) ? true : false;

        is(SecItemCopyMatching((CFTypeRef)query, NULL), multiUser ? errSecItemNotFound : noErr, "Bluetooth item - user keychain");

        query = @{
                  (__bridge id)kSecClass : (__bridge id)kSecClassGenericPassword,
                  (__bridge id)kSecAttrService : kBlueToothServiceName,
                  (__bridge id)kSecUseSystemKeychain : @YES,
                  };
        
        is(SecItemCopyMatching((CFTypeRef)query, NULL), noErr, "Bluetooth item - system keychain");
    }
}

int si_17_item_system_bluetooth(int argc, char *const *argv)
{
    plan_tests(2);

    @autoreleasepool {
        tests();
    }
    
    return 0;
}
