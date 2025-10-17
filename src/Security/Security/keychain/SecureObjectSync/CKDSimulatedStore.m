/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
//  CKDSimulatedStore.m
//  Security
//

#import "CKDSimulatedStore.h"
#import "CKDKVSProxy.h"

#include "SOSCloudKeychainConstants.h"
#include <utilities/debugging.h>

#import "SyncedDefaults/SYDConstants.h"
#include <os/activity.h>

@interface CKDSimulatedStore ()
@property (readwrite, weak) UbiqitousKVSProxy* proxy;
@property (readwrite) NSMutableDictionary<NSString*, NSObject*>* data;
@end

@implementation CKDSimulatedStore

+ (instancetype)simulatedInterface {
    return [[CKDSimulatedStore alloc] init];
}

- (instancetype)init {
    if ((self = [super init])) {
        self.proxy = nil;
        self.data = [NSMutableDictionary<NSString*, NSObject*> dictionary];
    }
    return self;
}

- (void) connectToProxy: (UbiqitousKVSProxy*) proxy {
    _proxy = proxy;
}

- (void)setObject:(id)obj forKey:(NSString*)key {
    [self.data setValue: obj forKey: key];
}

- (NSDictionary<NSString *, id>*) copyAsDictionary {
    return self.data;
}

- (void)addEntriesFromDictionary:(NSDictionary<NSString*, NSObject*> *)otherDictionary {
    [otherDictionary enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSObject * _Nonnull obj, BOOL * _Nonnull stop) {
        [self setObject:obj forKey:key];
    }];
}

- (id)objectForKey:(NSString*)key {
    return [self.data objectForKey:key];
}

- (void)removeObjectForKey:(NSString*)key {
    return [self.data removeObjectForKey:key];
}

- (void)removeAllObjects {
    [self.data removeAllObjects];
}

- (void)pushWrites:(NSArray<NSString*>*)keys requiresForceSync:(BOOL)requiresForceSync
{
}

- (void)addOneToOutGoing{
    
}
- (BOOL) pullUpdates:(NSError **)failure
{
    return true;
}

- (void)remoteSetObject:(id)obj forKey:(NSString*)key
{
    [self.data setObject:obj forKey:key];

    [self.proxy storeKeysChanged: [NSSet setWithObject:key] initial: NO];
}

- (void)perfCounters:(void(^)(NSDictionary *counters))callback
{
    callback(@{});
}

@end
