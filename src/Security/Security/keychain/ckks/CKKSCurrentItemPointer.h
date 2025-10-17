/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#import "keychain/ckks/CKKSItem.h"
#import "keychain/ckks/CKKSKey.h"

#if OCTAGON

NS_ASSUME_NONNULL_BEGIN

@interface CKKSCurrentItemPointer : CKKSCKRecordHolder

@property CKKSProcessedState* state;
@property NSString* identifier;
@property NSString* currentItemUUID;

- (instancetype)initForIdentifier:(NSString*)identifier
                        contextID:(NSString*)contextID
                  currentItemUUID:(NSString*)currentItemUUID
                            state:(CKKSProcessedState*)state
                           zoneID:(CKRecordZoneID*)zoneID
                  encodedCKRecord:(NSData* _Nullable)encodedrecord;

+ (instancetype)fromDatabase:(NSString*)identifier
                   contextID:(NSString*)contextID
                       state:(CKKSProcessedState*)state
                      zoneID:(CKRecordZoneID*)zoneID
                       error:(NSError* __autoreleasing*)error;
+ (instancetype)tryFromDatabase:(NSString*)identifier
                      contextID:(NSString*)contextID
                          state:(CKKSProcessedState*)state
                         zoneID:(CKRecordZoneID*)zoneID
                          error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSCurrentItemPointer*>*)remoteItemPointers:(CKRecordZoneID*)zoneID
                                              contextID:(NSString*)contextID
                                                  error:(NSError* __autoreleasing*)error;

+ (bool)deleteAll:(CKRecordZoneID*)zoneID
        contextID:(NSString*)contextID
            error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSCurrentItemPointer*>*)allInZone:(CKRecordZoneID*)zoneID
                                     contextID:(NSString*)contextID
                                         error:(NSError* __autoreleasing*)error;

+ (NSInteger)countByState:(CKKSItemState *)state
                contextID:(NSString*)contextID
                     zone:(CKRecordZoneID*)zoneID
                    error:(NSError * __autoreleasing *)error;

+ (BOOL)intransactionRecordChanged:(CKRecord*)record
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;

+ (BOOL)intransactionRecordDeleted:(CKRecordID*)recordID
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END

#endif
