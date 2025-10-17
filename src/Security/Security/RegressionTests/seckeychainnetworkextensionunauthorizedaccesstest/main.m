/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
//  main.m
//  seckeychainnetworkextensionunauthorizedaccesstest
//
//  Created by Luke Hiesterman on 2/23/17.
//

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecItemPriv.h>
#import <err.h>

static NSString* NetworkExtensionPersistentRefSharingAccessGroup = @"com.apple.NetworkExtensionPersistentRefSharingAccessGroup";
static NSString* TestAccount = @"MyTestAccount";

int main(int argc, const char* argv[])
{
    @autoreleasepool {
        NSMutableDictionary* attributes = [NSMutableDictionary dictionary];
        attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
        attributes[(__bridge NSString*)kSecAttrAccessGroup] = NetworkExtensionPersistentRefSharingAccessGroup;
        attributes[(__bridge NSString*)kSecAttrAccount] = TestAccount;
        attributes[(__bridge NSString*)kSecReturnData] = @YES;
        attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;

        CFTypeRef persistentRefData = NULL;
        OSStatus result = SecItemCopyMatching((__bridge CFDictionaryRef)attributes, &persistentRefData);
        if (result != 0 || !persistentRefData) {
            NSLog(@"got an error: %d", (int)result);
            errx(1, "failed to retrieve persistent ref data from keychain");
        }
        
        attributes = [NSMutableDictionary dictionary];
        attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
        attributes[(__bridge NSString*)kSecValuePersistentRef] = (__bridge NSData*)persistentRefData;
        attributes[(__bridge NSString*)kSecReturnData] = @YES;
        attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;

        CFTypeRef passwordData = NULL;
        result = SecItemCopyMatching((__bridge CFDictionaryRef)attributes, &passwordData);
        if (result == 0 && passwordData) {
            errx(1, "failed to block unentitled app from accessing network extension keychain item");
        }
        else {
            NSLog(@"successfully blocked unentitled app from accessing network extension keychain item");
        }
    }
    return 0;
}
