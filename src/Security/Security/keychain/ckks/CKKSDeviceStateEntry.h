/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
#ifndef CKKSDeviceStateEntry_h
#define CKKSDeviceStateEntry_h

#if OCTAGON

#include "keychain/securityd/SecDbItem.h"
#include "utilities/SecDb.h"

#import <CloudKit/CloudKit.h>
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSRecordHolder.h"
#import "keychain/ckks/CKKSAccountStateTracker.h"
#import "keychain/ckks/CKKSSQLDatabaseObject.h"

#import "keychain/ot/OTClique.h"

NS_ASSUME_NONNULL_BEGIN

/*
 * This is the backing class for "device state" records: each device in an iCloud account copies
 * some state about itself into each keychain view it wants to participate in.
 *
 * This shares some overlap with the CKKSZoneStateEntry, but differs in that:
 *   - This will be uploaded to CloudKit
 *   - We will have receive such records from other devices
 */

@class CKKSKeychainViewState;

@interface CKKSDeviceStateEntry : CKKSCKRecordHolder

@property NSString* device;

@property (nullable) NSString* osVersion;
@property (nullable) NSDate* lastUnlockTime;

@property (nullable) NSString* circlePeerID;
@property (nullable) NSString* octagonPeerID;

@property SOSCCStatus circleStatus;

// Some devices don't have Octagon, and won't upload this. Therefore, it might not be present,
// and I'd rather not coerce to "error" or "absent"
@property (nullable) OTCliqueStatusWrapper* octagonStatus;

@property (nullable) CKKSZoneKeyState* keyState;

@property (nullable) NSString* currentTLKUUID;
@property (nullable) NSString* currentClassAUUID;
@property (nullable) NSString* currentClassCUUID;

+ (instancetype)fromDatabase:(NSString*)device
                   contextID:(NSString*)contextID
                      zoneID:(CKRecordZoneID*)zoneID
                       error:(NSError* __autoreleasing*)error;

+ (instancetype)tryFromDatabase:(NSString*)device
                      contextID:(NSString*)contextID
                         zoneID:(CKRecordZoneID*)zoneID
                          error:(NSError* __autoreleasing*)error;

+ (instancetype)tryFromDatabaseFromCKRecordID:(CKRecordID*)recordID
                                    contextID:(NSString*)contextID
                                        error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSDeviceStateEntry*>*)allInZone:(CKRecordZoneID*)zoneID
                                       error:(NSError* __autoreleasing*)error;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initForDevice:(NSString*)device
                    contextID:(NSString*)contextID
                    osVersion:(NSString* _Nullable)osVersion
               lastUnlockTime:(NSDate* _Nullable)lastUnlockTime
                octagonPeerID:(NSString* _Nullable)octagonPeerID
                octagonStatus:(OTCliqueStatusWrapper* _Nullable)octagonStatus
                 circlePeerID:(NSString* _Nullable)circlePeerID
                 circleStatus:(SOSCCStatus)circleStatus
                     keyState:(CKKSZoneKeyState* _Nullable)keyState
               currentTLKUUID:(NSString* _Nullable)currentTLKUUID
            currentClassAUUID:(NSString* _Nullable)currentClassAUUID
            currentClassCUUID:(NSString* _Nullable)currentClassCUUID
                       zoneID:(CKRecordZoneID*)zoneID
              encodedCKRecord:(NSData* _Nullable)encodedrecord;

+ (CKKSDeviceStateEntry* _Nullable)intransactionCreateDeviceStateForView:(CKKSKeychainViewState*)viewState
                                                          accountTracker:(CKKSAccountStateTracker*)accountTracker
                                                        lockStateTracker:(CKKSLockStateTracker*)lockStateTracker
                                                                   error:(NSError**)error;

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

#endif  // OCTAGON
#endif  /* CKKSDeviceStateEntry_h */
