/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include "keychain/securityd/SecDbItem.h"
#include "utilities/SecDb.h"
#import "CKKSItem.h"
#import "CKKSSQLDatabaseObject.h"

#ifndef CKKSMirrorEntry_h
#define CKKSMirrorEntry_h

#import <CloudKit/CloudKit.h>

NS_ASSUME_NONNULL_BEGIN

@class CKKSWrappedAESSIVKey;

@interface CKKSMirrorEntry : CKKSSQLDatabaseObject

@property CKKSItem* item;
@property NSString* uuid;

@property uint64_t wasCurrent;

- (instancetype)initWithCKKSItem:(CKKSItem*)item;
- (instancetype)initWithCKRecord:(CKRecord*)record
                       contextID:(NSString*)contextID;
- (void)setFromCKRecord:(CKRecord*)record;
- (bool)matchesCKRecord:(CKRecord*)record;
- (bool)matchesCKRecord:(CKRecord*)record checkServerFields:(bool)checkServerFields;

+ (instancetype _Nullable)fromDatabase:(NSString*)uuid
                             contextID:(NSString*)contextID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError * __autoreleasing *)error;
+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError * __autoreleasing *)error;

+ (NSArray<CKKSMirrorEntry*>*)allWithUUID:(NSString*)uuid
                                contextID:(NSString*)contextID
                                    error:(NSError**)error;

+ (NSDictionary<NSString*,NSNumber*>*)countsByParentKeyWithContextID:(NSString*)contextID
                                                              zoneID:(CKRecordZoneID*)zoneID
                                                               error:(NSError * __autoreleasing *)error;
+ (NSNumber* _Nullable)countsWithContextID:(NSString*)contextID
                                    zoneID:(CKRecordZoneID*)zoneID
                                     error:(NSError * __autoreleasing *)error;

+ (NSDictionary<NSString*,NSNumber*>*)countsByZoneNameWithContextID:(NSString*)contextID
                                                              error:(NSError * __autoreleasing *)error;

+ (NSArray<NSData*>*)pcsMirrorKeysForService:(NSNumber*)service
                                matchingKeys:(NSArray<NSData*>*)matchingKeys
                                       error:(NSError * __autoreleasing *)error;

@end

NS_ASSUME_NONNULL_END
#endif
#endif /* CKKSOutgoingQueueEntry_h */
