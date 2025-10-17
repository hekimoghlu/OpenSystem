/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
//  Keychain.h
//  Security
//
//  Created by Ben Williamson on 6/2/17.
//
//

#import <Foundation/Foundation.h>
#import <Security/Security.h>

@interface Keychain : NSObject

// Should return errSecSuccess or errSecDuplicateItem
- (OSStatus)addItem:(NSString *)name value:(NSString *)value view:(NSString *)view;
- (OSStatus)addItem:(NSString *)name value:(NSString *)value view:(NSString *)view pRef:(NSArray **)result;

// Should return errSecSuccess or errSecItemNotFound
- (OSStatus)updateItemWithName:(NSString *)name newValue:(NSString *)newValue;
- (OSStatus)updateItem:(id)pRef newValue:(NSString *)newValue;
- (OSStatus)updateItem:(id)pRef newName:(NSString *)newName;
- (OSStatus)updateItem:(id)pRef newName:(NSString *)newName newValue:(NSString *)newValue;
- (OSStatus)deleteItem:(id)pRef;
- (OSStatus)deleteItemWithName:(NSString *)name;

// Should return errSecSuccess
- (OSStatus)deleteAllItems;

- (NSDictionary<NSString *, NSArray *> *)getAllItems;

@end
