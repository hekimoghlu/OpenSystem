/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include "keychain/securityd/SecDbItem.h"
#include "utilities/SecDb.h"
#import "CKKSItem.h"
#import "CKKSMirrorEntry.h"
#import "CKKSSQLDatabaseObject.h"

#ifndef CKKSOutgoingQueueEntry_h
#define CKKSOutgoingQueueEntry_h

#if OCTAGON
#import <CloudKit/CloudKit.h>
#import "keychain/ckks/CKKSMemoryKeyCache.h"

NS_ASSUME_NONNULL_BEGIN

@class CKKSKeychainView;
@class CKKSKeychainViewState;

@interface CKKSOutgoingQueueEntry : CKKSSQLDatabaseObject

@property CKKSItem* item;
@property NSString* uuid;  // property access to underlying CKKSItem

@property NSString* action;
@property NSString* state;
@property (readonly) NSString* contextID;
@property NSString* accessgroup;
@property NSDate* waitUntil;  // If non-null, the time at which this entry should be processed

- (instancetype)initWithCKKSItem:(CKKSItem*)item
                          action:(NSString*)action
                           state:(NSString*)state
                       waitUntil:(NSDate* _Nullable)waitUntil
                     accessGroup:(NSString*)accessgroup;

+ (instancetype _Nullable)withItem:(SecDbItemRef)item
                            action:(NSString*)action
                         contextID:(NSString*)contextID
                            zoneID:(CKRecordZoneID*)zoneID
                          keyCache:(CKKSMemoryKeyCache* _Nullable)keyCache
                             error:(NSError * __autoreleasing *)error;

+ (instancetype _Nullable)fromDatabase:(NSString*)uuid
                                 state:(NSString*)state
                             contextID:(NSString*)contextID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError* __autoreleasing*)error;
+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError* __autoreleasing*)error;
+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                    state:(NSString*)state
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSOutgoingQueueEntry*>*)fetch:(ssize_t)n
                                     state:(NSString*)state
                                 contextID:(NSString*)contextID
                                    zoneID:(CKRecordZoneID*)zoneID
                                     error:(NSError* __autoreleasing*)error;
+ (NSArray<CKKSOutgoingQueueEntry*>*)allInState:(NSString*)state
                                      contextID:(NSString*)contextID
                                         zoneID:(CKRecordZoneID*)zoneID
                                          error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSOutgoingQueueEntry*>*)allWithUUID:(NSString*)uuid
                                          states:(NSArray<NSString*>*)states
                                       contextID:(NSString*)contextID
                                          zoneID:(CKRecordZoneID*)zoneID
                                           error:(NSError * __autoreleasing *)error;

+ (NSDictionary<NSString*, NSNumber*>*)countsByStateWithContextID:(NSString*)contextID
                                                           zoneID:(CKRecordZoneID*)zoneID
                                                            error:(NSError* __autoreleasing*)error;
+ (NSInteger)countByState:(CKKSItemState *)state
                contextID:(NSString*)contextID
                   zoneID:(CKRecordZoneID*)zoneID
                    error:(NSError * __autoreleasing *)error;

- (BOOL)intransactionMoveToState:(NSString*)state
                       viewState:(CKKSKeychainViewState*)viewState
                           error:(NSError**)error;
- (BOOL)intransactionMarkAsError:(NSError*)itemError
                       viewState:(CKKSKeychainViewState*)viewState
                           error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
#endif
#endif /* CKKSOutgoingQueueEntry_h */
