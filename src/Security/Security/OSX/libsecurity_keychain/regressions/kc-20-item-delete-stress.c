/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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

#include "keychain_regressions.h"
#include "kc-helpers.h"
#include "kc-item-helpers.h"

#include <dispatch/private.h>
#include <Security/Security.h>

#include <stdlib.h>

#define BLOCKS 30

static OSStatus callbackFunction(SecKeychainEvent keychainEvent,
                                 SecKeychainCallbackInfo *info, void *context)
{
    CFRetainSafe(info->item);
    //printf("received a callback: %d %p\n", keychainEvent, info->item);
    CFReleaseNull(info->item);

    return 0;
}

static void tests(void) {
    SecKeychainRef kc = getEmptyTestKeychain();

    static dispatch_once_t onceToken = 0;
    static dispatch_queue_t process_queue = NULL;
    dispatch_once(&onceToken, ^{
        process_queue = dispatch_queue_create("com.apple.security.item-add-queue", DISPATCH_QUEUE_CONCURRENT);

        dispatch_queue_set_width(process_queue, 40);
    });
    dispatch_group_t g = dispatch_group_create();


    // Run the CFRunLoop to clear out existing notifications
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, false);

    UInt32 didGetNotification = 0;
    ok_status(SecKeychainAddCallback(callbackFunction, kSecAddEventMask | kSecDeleteEventMask, &didGetNotification), "add callback");

    // Run the CFRunLoop to mark this run loop as "pumped"
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, false);

    for(int i = 0; i < BLOCKS; i++) {
        dispatch_group_async(g, process_queue, ^{
            SecKeychainItemRef blockItem = NULL;
            CFStringRef itemclass = kSecClassInternetPassword;

            CFStringRef label = CFStringCreateWithFormat(NULL, NULL, CFSTR("testItem%05d"), i);
            CFStringRef account = CFSTR("testAccount");
            CFStringRef service = CFStringCreateWithFormat(NULL, NULL, CFSTR("testService%05d"), i);
            char * name;
            asprintf(&name, "%s (item %d)", testName, i);

            // add the item
            blockItem = createCustomItem(name, kc, createAddCustomItemDictionaryWithService(kc, itemclass, label, account, service));

            ok_status(SecKeychainItemDelete(blockItem), "%s: SecKeychainItemDelete", name);
            usleep(100 * arc4random_uniform(10000));
            CFReleaseNull(blockItem);

            free(name);
            CFReleaseNull(label);
            CFReleaseNull(service);
        });
    }

    // Process run loop until every block has run
    while(dispatch_group_wait(g, DISPATCH_TIME_NOW) != 0) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, false);
    }

    // One last hurrah
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, false);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);
}

int kc_20_item_delete_stress(int argc, char *const *argv)
{
    plan_tests(getEmptyTestKeychainTests + 1 + (createCustomItemTests + 1)*BLOCKS + 1);
    initializeKeychainTests(__FUNCTION__);
    
    tests();
    
    deleteTestFiles();
    return 0;
}
