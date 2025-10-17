/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#if OCTAGON

#import <CloudKit/CloudKit.h>
#include "keychain/securityd/SecDbItem.h"
#include "utilities/SecDb.h"
#import "CKKSItem.h"
#import "CKKSMirrorEntry.h"
#import "CKKSSQLDatabaseObject.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSIncomingQueueEntry : CKKSSQLDatabaseObject

@property CKKSItem* item;
@property NSString* uuid;  // through-access to underlying item

@property NSString* action;
@property NSString* state;
@property (readonly) NSString* contextID;

- (instancetype)initWithCKKSItem:(CKKSItem*)ckme action:(NSString*)action state:(NSString*)state;

+ (instancetype _Nullable)fromDatabase:(NSString*)uuid
                             contextID:(NSString*)contextID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError* __autoreleasing*)error;
+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSIncomingQueueEntry*>* _Nullable)fetch:(ssize_t)n
                                      startingAtUUID:(NSString* _Nullable)uuid
                                               state:(NSString*)state
                                              action:(NSString* _Nullable)action
                                           contextID:(NSString*)contextID
                                              zoneID:(CKRecordZoneID*)zoneID
                                               error:(NSError* __autoreleasing*)error;

+ (NSDictionary<NSString*, NSNumber*>*)countsByStateWithContextID:(NSString*)contextID
                                                           zoneID:(CKRecordZoneID*)zoneID
                                                            error:(NSError* __autoreleasing*)error;
+ (NSInteger)countByState:(CKKSItemState *)state
                contextID:(NSString*)contextID
                     zone:(CKRecordZoneID*)zoneID
                    error:(NSError * __autoreleasing *)error;

+ (NSDictionary<NSString*, NSNumber*>*)countNewEntriesByKeyWithContextID:(NSString*)contextID
                                                                  zoneID:(CKRecordZoneID*)zoneID
                                                                   error:(NSError* __autoreleasing*)error;

// Returns true if all extant IQEs for the given zone have parent keys which exist and can be loaded (whether or not they're local or reoote)
// This is intended to return false if CKKS desyncs from the server about the existence of a sync key
+ (BOOL)allIQEsHaveValidUnwrappingKeysInContextID:(NSString*)contextID
                                           zoneID:(CKRecordZoneID*)zoneID
                                            error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
#endif
