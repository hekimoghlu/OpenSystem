/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
//  CKDSimulatedStore.h
//  Security
//
//  Created by Mitch Adler on 10/31/16.
//
//

#import "CKDStore.h"

@interface CKDSimulatedStore : NSObject <CKDStore>

+ (instancetype)simulatedInterface;
- (instancetype)init;

- (void)connectToProxy: (UbiqitousKVSProxy*) proxy;

- (NSObject*)objectForKey:(NSString*)key;

- (void)setObject:(id)obj forKey:(NSString*)key;
- (void)addEntriesFromDictionary:(NSDictionary<NSString*, NSObject*> *)otherDictionary;

- (void)removeObjectForKey:(NSString*)key;
- (void)removeAllObjects;

- (NSDictionary<NSString *, id>*) copyAsDictionary;

- (void)pushWrites:(NSArray<NSString*>*)keys requiresForceSync:(BOOL)requiresForceSync;
- (BOOL)pullUpdates:(NSError**) failure;
- (void)addOneToOutGoing;


- (void)remoteSetObject:(id)obj forKey:(NSString*)key;

@end
