/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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

#import "keychain/ckks/CKKSKey.h"

#if OCTAGON

NS_ASSUME_NONNULL_BEGIN

//
// Usage Note:
//
//  This object transparently caches CKKSKey objects, with their key material
//  intact. Since those keys are loaded from the database, they remain
//  valid while you're in the database transaction where you loaded them.
//  To preverve this property, this cache must be destroyed before the end
//  of the database transaction in which it is created.
//

@interface CKKSMemoryKeyCache : NSObject

// An instance of a CKKSItemEncrypter also contains a cache of (loaded and ready) CKKSKeys
// Use these to access the cache
- (instancetype)init;
- (CKKSKey* _Nullable)loadKeyForUUID:(NSString*)keyUUID
                           contextID:(NSString*)contextID
                              zoneID:(CKRecordZoneID*)zoneID
                               error:(NSError**)error;
- (CKKSKey* _Nullable)currentKeyForClass:(CKKSKeyClass*)keyclass
                               contextID:(NSString*)contextID
                                  zoneID:(CKRecordZoneID*)zoneID
                                   error:(NSError *__autoreleasing*)error;
- (void)addKeyToCache:(NSString*)keyUUID
                  key:(CKKSKey*)key;

- (void)populateWithRecords:(NSArray<CKRecord*>*)syncKeys
                  contextID:(NSString*)contextID;

@end

NS_ASSUME_NONNULL_END

#endif
