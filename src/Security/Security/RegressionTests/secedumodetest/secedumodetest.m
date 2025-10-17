/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
//  Copyright 2016 Apple. All rights reserved.
//

#include <Foundation/Foundation.h>
#include <Security/Security.h>
#include <Security/SecItemPriv.h>
#include <TargetConditionals.h>
#include <err.h>


int main (int argc, const char * argv[])
{
    @autoreleasepool {

        NSDictionary *addItem = @{
              (id)kSecClass : (id)kSecClassGenericPassword,
              (id)kSecAttrLabel : @"vpn-test-label",
              (id)kSecAttrAccount : @"vpn-test-account",
              (id)kSecValueData : @"password",
              (id)kSecUseSystemKeychain : @YES,
        };

        NSDictionary *querySystemItem = @{
            (id)kSecClass : (id)kSecClassGenericPassword,
            (id)kSecAttrLabel : @"vpn-test-label",
            (id)kSecAttrAccount : @"vpn-test-account",
            (id)kSecUseSystemKeychain : @YES,
        };

        NSDictionary *queryItem = @{
            (id)kSecClass : (id)kSecClassGenericPassword,
            (id)kSecAttrLabel : @"vpn-test-label",
            (id)kSecAttrAccount : @"vpn-test-account",
        };

        (void)SecItemDelete((__bridge CFDictionaryRef)querySystemItem);


        if (!SecItemAdd((__bridge CFDictionaryRef)addItem, NULL))
            errx(1, "failed to add");


        if (!SecItemCopyMatching((__bridge CFDictionaryRef)queryItem, NULL))
            errx(1, "failed to find in user + system");

        if (!SecItemCopyMatching((__bridge CFDictionaryRef)querySystemItem, NULL))
            errx(1, "failed to find in system");


        if (!SecItemDelete((__bridge CFDictionaryRef)querySystemItem))
            errx(1, "failed to clean up");

        return 0;
    }
}


