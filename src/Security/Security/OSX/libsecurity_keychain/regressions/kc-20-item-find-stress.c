/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#import <Security/Security.h>
#import <Security/SecCertificatePriv.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"
#include "kc-item-helpers.h"

#include <Security/Security.h>
#include <stdlib.h>

#define BLOCKS 7000

static long concurrentBlocks = 64;

static void tests(void) {

    SecKeychainRef kc = getPopulatedTestKeychain();

    static dispatch_once_t onceToken = 0;
    static dispatch_queue_t release_queue = NULL;
    dispatch_once(&onceToken, ^{
        release_queue = dispatch_queue_create("com.apple.security.identity-search-queue", DISPATCH_QUEUE_CONCURRENT);
    });
    dispatch_group_t g = dispatch_group_create();

    __block int iteration = 0;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(concurrentBlocks);

    dispatch_block_t findBlock = ^{
        SecKeychainItemRef blockItem = NULL;

        CFMutableDictionaryRef query = createQueryCustomItemDictionaryWithService(kc, kSecClassInternetPassword, CFSTR("test_service"), CFSTR("test_service"));
        CFDictionarySetValue(query, kSecMatchLimit, kSecMatchLimitOne);

        ok_status(SecItemCopyMatching(query, (CFTypeRef*) &blockItem), "%s: SecItemCopyMatching", testName);
        CFReleaseNull(query);

        CFReleaseNull(blockItem);
        dispatch_semaphore_signal(semaphore);
    };

    // Send this to background queue, so that when we wait, it doesn't block main queue

    dispatch_group_async(g, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        for (iteration = 0; iteration < BLOCKS; iteration++) {
            // Wait until one of our concurrentBlocks "slots" are available
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

            dispatch_group_async(g, release_queue, findBlock);
        }
    });

    dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);
}

int kc_20_item_find_stress(int argc, char *const *argv)
{
    plan_tests((1)*BLOCKS + getPopulatedTestKeychainTests + 1);
    initializeKeychainTests(__FUNCTION__);

    tests();

    deleteTestFiles();
    return 0;
}
