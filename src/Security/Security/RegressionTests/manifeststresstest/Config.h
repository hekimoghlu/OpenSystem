/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
//  Config.h
//  Security
//
//  Created by Ben Williamson on 6/2/17.
//
//

#import <Foundation/Foundation.h>

@interface Config : NSObject

// Number of distinct item names to chose from.
@property (nonatomic, assign) unsigned distinctNames;

// Number of distinct data values to chose from.
@property (nonatomic, assign) unsigned distinctValues;

// Max number of items we are allowed to create.
@property (nonatomic, assign) unsigned maxItems;

// Probability weighting for adding an item.
@property (nonatomic, assign) unsigned addItemWeight;

// Probability weighting for updating an item's name.
@property (nonatomic, assign) unsigned updateNameWeight;

// Probability weighting for updating an item's data.
@property (nonatomic, assign) unsigned updateDataWeight;

// Probability weighting for updating an item's name and data.
@property (nonatomic, assign) unsigned updateNameAndDataWeight;

// Probability weighting for deleting an item.
@property (nonatomic, assign) unsigned deleteItemWeight;

// Additional item name configuration, for isolating changes
@property (nonatomic) NSString *name;

// Additional view name
@property (nonatomic) NSString *view;

@end
