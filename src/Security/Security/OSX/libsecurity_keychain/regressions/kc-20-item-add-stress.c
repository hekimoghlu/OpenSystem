/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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

#include <Security/Security.h>

#define BLOCKS 300

static void tests(void) {
    SecKeychainRef kc = getPopulatedTestKeychain();

    static dispatch_once_t onceToken = 0;
    static dispatch_queue_t process_queue = NULL;
    dispatch_once(&onceToken, ^{
        process_queue = dispatch_queue_create("com.apple.security.item-add-queue", DISPATCH_QUEUE_CONCURRENT);
    });
    dispatch_group_t g = dispatch_group_create();

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
            CFReleaseNull(blockItem);

            // find the item
            blockItem = checkNCopyFirst(name, createQueryCustomItemDictionaryWithService(kc, itemclass, label, service), 1);
            readPasswordContents(blockItem, CFSTR("data"));

            // update the item
            CFMutableDictionaryRef newData = CFDictionaryCreateMutable(NULL, 0,
                                                                     &kCFTypeDictionaryKeyCallBacks,
                                                                     &kCFTypeDictionaryValueCallBacks);
            CFDataRef differentdata = CFDataCreate(NULL, (void*)"differentdata", strlen("differentdata"));
            CFDictionarySetValue(newData, kSecValueData, differentdata);
            CFReleaseNull(differentdata);

            CFDictionaryRef query = createQueryCustomItemDictionaryWithService(kc, itemclass, label, service);
            ok_status(SecItemUpdate(query, newData), "%s: SecItemUpdate", name);
            CFReleaseNull(query);
            readPasswordContents(blockItem, CFSTR("differentdata"));

            // delete the item
            ok_status(SecKeychainItemDelete(blockItem), "%s: SecKeychainItemDelete", name);
            CFReleaseNull(blockItem);
            blockItem = checkNCopyFirst(name, createQueryCustomItemDictionaryWithService(kc, itemclass, label, service), 0);

            free(name);
            CFReleaseNull(label);
            CFReleaseNull(service);
            CFReleaseNull(blockItem);
        });
    }

    dispatch_group_wait(g, DISPATCH_TIME_FOREVER);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);
}

int kc_20_item_add_stress(int argc, char *const *argv)
{
    plan_tests(( makeItemTests + checkNTests + readPasswordContentsTests + 1 + readPasswordContentsTests + 1 + checkNTests )*BLOCKS + getPopulatedTestKeychainTests + 1);
    initializeKeychainTests(__FUNCTION__);

    tests();

    deleteTestFiles();
    return 0;
}
