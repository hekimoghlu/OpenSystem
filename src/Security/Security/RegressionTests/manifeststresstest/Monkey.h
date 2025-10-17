/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
//  Monkey.h
//  Security
//
//  Created by Ben Williamson on 6/1/17.
//
//

#import <Foundation/Foundation.h>

@class Config;
@class Keychain;

// A monkey has an array of items that it has created.
// It can randomly choose to add an item, delete an item, or update the data in an item.
//
// All items exist within the access group "manifeststresstest"
// which is set to have the appropriate view hint so that it syncs via CKKS.
//
// Items are generic password items, having a service, an account and data.
// The service, account and data values are chosen from sets of a limited size, to encourage
// the possibility of collisions.


// Adds and deletes generic password items

@interface Monkey : NSObject

@property (nonatomic, strong) Keychain *keychain; // if nil, this is a dry run
@property (nonatomic, assign) unsigned step;

// Incremented when we try to add an item and it already exists.
@property (nonatomic, assign) unsigned addDuplicateCounter;

// Incremented when we try to update or delete an item and it does not exist.
@property (nonatomic, assign) unsigned notFoundCounter;

// Peak number of items we have created so far.
@property (nonatomic, assign) unsigned peakItems;

// Current number of items written
@property (nonatomic, readonly) unsigned itemCount;

@property (nonatomic, readonly) Config *config;

- (instancetype)initWithConfig:(Config *)config;

- (void)advanceOneStep;

- (void)cleanup;

@end
