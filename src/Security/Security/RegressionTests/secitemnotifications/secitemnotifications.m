/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include <Security/Security.h>
#include <Security/SecItemPriv.h>
#include <notify.h>
#include <err.h>
#import <TargetConditionals.h>
#include "keychain/securityd/SOSCloudCircleServer.h"
#include "utilities/SecCFRelease.h"

int
main(int argc, const char ** argv)
{
    dispatch_queue_t queue = dispatch_queue_create("notifications-queue", NULL);
    __block int got_notification = false;

    OSStatus status;
    int token;

    NSDictionary *query = @{
        (id)kSecClass : (id)kSecClassGenericPassword,
        (id)kSecAttrAccessGroup : @"keychain-test1",
        (id)kSecAttrSyncViewHint : @"PCS-MasterKey",
        (id)kSecAttrAccount : @"account-delete-me",
        (id)kSecAttrSynchronizable : (id)kCFBooleanTrue,
        (id)kSecAttrAccessible : (id)kSecAttrAccessibleAfterFirstUnlock,
    };
    status = SecItemDelete((__bridge CFDictionaryRef)query);
    if (status != errSecSuccess && status != errSecItemNotFound) {
        errx(1, "cleanup item: %d", (int)status);
    }

    notify_register_dispatch("com.apple.security.view-change.PCS", &token, queue, ^(int __unused token2) {
        printf("got notification\n");
        got_notification = true;
    });

    /*
     * now check add notification
     */

    status = SecItemAdd((__bridge CFDictionaryRef)query, NULL);
    if (status != errSecSuccess) {
        errx(1, "add item: %d", (int)status);
    }

    sleep(3);

// Bridge explicitly disables notify phase, no PCS, octagon or sos on this platform
#if !TARGET_OS_BRIDGE
    if (!got_notification) {
        errx(1, "failed to get notification on add");
    }
#else
    if (got_notification) {
        errx(1, "received unexpected notification on add");
    }
#endif
    got_notification = false;

    /*
     * clean up and check delete notification too
     */

    status = SecItemDelete((__bridge CFDictionaryRef)query);
    if (status != errSecSuccess) {
        errx(1, "cleanup2 item: %d", (int)status);
    }

    sleep(3);

#if !TARGET_OS_BRIDGE
    if (!got_notification) {
        errx(1, "failed to get notification on delete");
    }
#else
    if (got_notification) {
        errx(1, "received unexpected notification on delete");
    }
#endif
    
    return 0;
}
