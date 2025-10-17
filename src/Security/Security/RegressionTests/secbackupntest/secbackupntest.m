/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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

//
//  Copyright 2015 Apple. All rights reserved.
//

#include <Foundation/Foundation.h>
#include <Security/Security.h>

#include <TargetConditionals.h>

#include <Security/SecItemPriv.h>
#include <err.h>

#if !TARGET_OS_SIMULATOR
#include <AppleKeyStore/libaks.h>

static NSData *
BagMe(void)
{
    keybag_handle_t handle;
    kern_return_t result;
    void *data = NULL;
    int length;

    result = aks_create_bag("foo", 3, kAppleKeyStoreAsymmetricBackupBag, &handle);
    if (result)
        errx(1, "aks_create_bag: %08x", result);

    result = aks_save_bag(handle, &data, &length);
    if (result)
        errx(1, "aks_save_bag");

    return [NSData dataWithBytes:data length:length];
}
#endif /* TARGET_OS_SIMULATOR */

int main (int argc, const char * argv[])
{
    @autoreleasepool {
        NSData *bag = NULL, *password = NULL;

#if !TARGET_OS_SIMULATOR
        bag = BagMe();
        password = [NSData dataWithBytes:"foo" length:3];
#endif

        NSLog(@"backup bag: %@", bag);

        NSData *backup = (__bridge NSData *)_SecKeychainCopyBackup((__bridge CFDataRef)bag, (__bridge CFDataRef)password);
        if (backup != NULL) {
            NSLog(@"backup data: %@", backup);
            errx(1, "got backup");
        }
        return 0;
    }
}


