/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#import <Foundation/Foundation.h>

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSMemoryKeyCache.h"
#import "keychain/ckks/CKKSCurrentKeyPointer.h"

#if OCTAGON

@interface CKKSMemoryKeyCache ()
@property NSMutableDictionary<NSString*, CKKSKey*>* keyCache;
@end

@implementation CKKSMemoryKeyCache

- (instancetype)init
{
    if((self = [super init])) {
        _keyCache = [NSMutableDictionary dictionary];
    }
    return self;
}

- (CKKSKey* _Nullable)loadKeyForUUID:(NSString*)keyUUID
                           contextID:(NSString*)contextID
                              zoneID:(CKRecordZoneID*)zoneID
                               error:(NSError**)error
{
    CKKSKey* key = self.keyCache[keyUUID];
    if(key) {
        return key;
    }

    // Note: returns nil (and empties the cache) if there is an error
    key = [CKKSKey loadKeyWithUUID:keyUUID
                         contextID:contextID
                            zoneID:zoneID
                             error:error];
    self.keyCache[keyUUID] = key;
    return key;
}

- (CKKSKey* _Nullable)loadKeyForItem:(CKKSItem*)item error:(NSError**)error
{
    return [self loadKeyForUUID:item.parentKeyUUID
                      contextID:item.contextID
                         zoneID:item.zoneID
                          error:error];
}

- (CKKSKey* _Nullable)currentKeyForClass:(CKKSKeyClass*)keyclass
                               contextID:(NSString*)contextID
                                  zoneID:(CKRecordZoneID*)zoneID
                                   error:(NSError *__autoreleasing*)error
{
    // Load the CurrentKey record, and find the key for it
    CKKSCurrentKeyPointer* ckp = [CKKSCurrentKeyPointer fromDatabase:keyclass
                                                           contextID:contextID
                                                              zoneID:zoneID
                                                               error:error];
    if(!ckp) {
        return nil;
    }
    return [self loadKeyForUUID:ckp.currentKeyUUID
                      contextID:contextID
                         zoneID:zoneID
                          error:error];
}

- (void)addKeyToCache:(NSString*)keyUUID
                  key:(CKKSKey*)key
{
    self.keyCache[keyUUID] = key;
}

- (void)populateWithRecords:(NSArray<CKRecord*>*)syncKeys
                  contextID:(NSString*)contextID {
    for (CKRecord* obj in syncKeys) {
        CKKSKey* key = [[CKKSKey alloc] initWithCKRecord:obj contextID:contextID];
        [self addKeyToCache:key.uuid key:key];
    }
}

@end

#endif
