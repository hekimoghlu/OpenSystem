/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
//  <rdar://3425797>

#include "keychain_regressions.h"
#include "kc-helpers.h"
#include "kc-item-helpers.h"

#import <Foundation/Foundation.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <CoreServices/CoreServices.h>

//Call SecKeychainAddGenericPassword to add a new password to the keychain:
static OSStatus StorePasswordKeychain (SecKeychainRef keychain, void* password,UInt32 passwordLength)
{
    OSStatus status;
    status = SecKeychainAddGenericPassword (
                                            keychain,
                                            10,                             // length of service name
                                            "SurfWriter",                   // service name
                                            10,                             // length of account name
                                            "MyUserAcct",                   // account name
                                            passwordLength,                 // length of password
                                            password,                       // pointer to password data
                                            NULL                            // the item reference
                                            );

    ok_status(status, "%s: SecKeychainAddGenericPassword", testName);
    return (status);
}

//Call SecKeychainFindGenericPassword to get a password from the keychain:
static OSStatus GetPasswordKeychain (SecKeychainRef keychain, void *passwordData,UInt32 *passwordLength,SecKeychainItemRef *itemRef)
{
    OSStatus status;


    status = SecKeychainFindGenericPassword (
                                            keychain,
                                            10,                             // length of service name
                                            "SurfWriter",                   // service name
                                            10,                             // length of account name
                                            "MyUserAcct",                   // account name
                                            passwordLength,                 // length of password
                                            passwordData,                   // pointer to password data
                                            itemRef                        // the item reference
                                            );
    ok_status(status, "%s: SecKeychainFindGenericPassword", testName);
    return (status);
}

//Call SecKeychainItemModifyAttributesAndData to change the password for an item already in the keychain:
static OSStatus ChangePasswordKeychain (SecKeychainItemRef itemRef)
{
    OSStatus status;
    void * password = "myNewP4sSw0rD";
    UInt32 passwordLength = (UInt32) strlen(password);
    void * label = "New Item Label";
    UInt32 labelLength =  (UInt32) strlen(label);

    NSString *account = @"New Account";
    NSString *service = @"New Service";
    const char *serviceUTF8 = [service UTF8String];
    const char *accountUTF8 = [account UTF8String];

//%%% IMPORTANT: While SecKeychainItemCreateFromContent() will accept a kSecLabelItemAttr, it cannot
// be changed later via SecKeychainItemModifyAttributesAndData(). ##### THIS IS A BUG. #####
// To work around the bug, pass 7 instead of kSecLabelItemAttr. This value is the index of the label
// attribute in the database schema (and in the SecItemAttr enumeration).
//
//#define LABEL_ITEM_ATTR_TAG 7
#define LABEL_ITEM_ATTR_TAG kSecLabelItemAttr

	// set up attribute vector (each attribute consists of {tag, length, pointer})
	SecKeychainAttribute attrs[] = {
        { LABEL_ITEM_ATTR_TAG, labelLength, (char *)label },
		{ kSecAccountItemAttr, (UInt32) strlen(accountUTF8), (char *)accountUTF8 },
		{ kSecServiceItemAttr, (UInt32) strlen(serviceUTF8), (char *)serviceUTF8 }	};
	const SecKeychainAttributeList attributes = { sizeof(attrs) / sizeof(attrs[0]), attrs };


    status = SecKeychainItemModifyAttributesAndData (
                                            itemRef,                        // the item reference
                                            &attributes,                    // attributes to change
                                            passwordLength,                 // length of password
                                            password                        // pointer to password data
                                            );
    ok_status(status, "%s: SecKeychainItemModifyAttributesAndData", testName);
    return (status);
}

int kc_15_item_update_label_skimaad(int argc, char *const *argv)
{
    plan_tests(28);
    initializeKeychainTests(__FUNCTION__);

    SecKeychainRef keychain = getPopulatedTestKeychain();

    OSStatus status;

    void * myPassword = "myP4sSw0rD";
    UInt32 myPasswordLength = (UInt32) strlen(myPassword);
    void *passwordData = nil; // will be allocated and filled in by SecKeychainFindGenericPassword
    UInt32 passwordLength = 0;
    SecKeychainItemRef itemRef = nil;

    StorePasswordKeychain(keychain, myPassword, myPasswordLength);

    itemRef = checkNCopyFirst(testName, createQueryCustomItemDictionaryWithService(keychain, kSecClassGenericPassword, CFSTR("SurfWriter"), CFSTR("SurfWriter")), 1);
    checkN(testName, createQueryCustomItemDictionaryWithService(keychain, kSecClassGenericPassword, CFSTR("New Item Label"), CFSTR("New Service")), 0);
    readPasswordContents(itemRef, CFSTR("myP4sSw0rD"));
    CFReleaseNull(itemRef);

    GetPasswordKeychain (keychain, &passwordData,&passwordLength,&itemRef);  //Call SecKeychainFindGenericPassword

    ok(passwordData, "Recieved password data");
    /*
     free the data allocated by SecKeychainFindGenericPassword:
     */
    status = SecKeychainItemFreeContent (
                                         NULL,                  //No attribute data to release
                                         passwordData          //Release data buffer allocated by SecKeychainFindGenericPassword
                                         );
    ok_status(status, "%s: SecKeychainItemFreeContent", testName);

    ChangePasswordKeychain(itemRef);

    checkN(testName, createQueryCustomItemDictionaryWithService(keychain, kSecClassGenericPassword, CFSTR("SurfWriter"), CFSTR("SurfWriter")), 0);
    itemRef = checkNCopyFirst(testName, createQueryCustomItemDictionaryWithService(keychain, kSecClassGenericPassword, CFSTR("New Item Label"), CFSTR("New Service")), 1);
    readPasswordContents(itemRef, CFSTR("myNewP4sSw0rD"));
    CFReleaseNull(itemRef);

    checkPrompts(0, "no prompts during test");

    ok_status(SecKeychainDelete(keychain), "%s: SecKeychainDelete", testName);
    CFReleaseNull(keychain);

    deleteTestFiles();
    return 0;
}
