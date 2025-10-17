/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
//  seckeychainnetworkextensionstest
//
//  Created by Luke Hiesterman on 2/22/17.
//

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecItemPriv.h>
#import <err.h>

static NSString* NetworkExtensionPersistentRefSharingAccessGroup = @"com.apple.NetworkExtensionPersistentRefSharingAccessGroup";
static NSString* NetworkExtensionAccessGroup = @"FakeAppPrefix.com.apple.networkextensionsharing";
static NSString* TestAccount = @"MyTestAccount";
static NSString* TestPassword = @"MyTestPassword";

static void cleanupKeychain(void)
{
    NSMutableDictionary* attributes = [NSMutableDictionary dictionary];
    attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
    attributes[(__bridge NSString*)kSecAttrAccessGroup] = NetworkExtensionAccessGroup;
    attributes[(__bridge NSString*)kSecAttrAccount] = TestAccount;
    attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;
    SecItemDelete((__bridge CFDictionaryRef)attributes);
    
    attributes = [NSMutableDictionary dictionary];
    attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
    attributes[(__bridge NSString*)kSecAttrAccessGroup] = NetworkExtensionPersistentRefSharingAccessGroup;
    attributes[(__bridge NSString*)kSecAttrAccount] = TestAccount;
    attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;
    SecItemDelete((__bridge CFDictionaryRef)attributes);
    
}

int main(int argc, const char * argv[])
{
    @autoreleasepool {
        cleanupKeychain();
        
        NSMutableDictionary* attributes = [NSMutableDictionary dictionary];
        attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
        attributes[(__bridge NSString*)kSecAttrAccessGroup] = NetworkExtensionAccessGroup;
        attributes[(__bridge NSString*)kSecAttrAccount] = TestAccount;
        attributes[(__bridge NSString*)kSecValueData] = [NSData dataWithBytes:TestPassword.UTF8String length:TestPassword.length];
        attributes[(__bridge NSString*)kSecReturnPersistentRef] = @YES;
        attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;

        CFTypeRef returnData = NULL;
        OSStatus result = SecItemAdd((__bridge CFDictionaryRef)attributes, &returnData);
        if (result != 0) {
            NSLog(@"got an error: %d", (int)result);
            errx(1, "failed to add item to keychain");
        }
        
        if (returnData) {
            attributes = [NSMutableDictionary dictionary];
            attributes[(__bridge NSString*)kSecClass] = (__bridge NSString*)kSecClassGenericPassword;
            attributes[(__bridge NSString*)kSecAttrAccessGroup] = NetworkExtensionPersistentRefSharingAccessGroup;
            attributes[(__bridge NSString*)kSecAttrAccount] = TestAccount;
            attributes[(__bridge NSString*)kSecValueData] = (__bridge NSData*)returnData;
            attributes[(__bridge NSString*)kSecUseDataProtectionKeychain] = @YES;

            result = SecItemAdd((__bridge CFDictionaryRef)attributes, &returnData);
            if (result == 0) {
                NSLog(@"successfully stored persistent ref for shared network extension item to keychain");
            }
            else {
                errx(1, "failed to add persistent ref to keychain");
            }
        }
        else {
            errx(1, "failed to get persistent ref from item added to keychain");
        }
    }
    return 0;
}
