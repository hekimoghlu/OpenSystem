/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#import "keychain/ckks/CKKSSQLDatabaseObject.h"

NS_ASSUME_NONNULL_BEGIN

// Helper class that includes a single encoded CKRecord
@interface CKKSCKRecordHolder : CKKSSQLDatabaseObject


- (instancetype)initWithCKRecord:(CKRecord*)record
                       contextID:(NSString*)contextID;
- (instancetype)initWithCKRecordType:(NSString*)recordType
                     encodedCKRecord:(NSData* _Nullable)encodedCKRecord
                           contextID:(NSString*)contextID
                              zoneID:(CKRecordZoneID*)zoneID;

@property (readonly) NSString* contextID;
@property CKRecordZoneID* zoneID;
@property NSString* ckRecordType;
@property (nullable, copy) NSData* encodedCKRecord;
@property (nullable, copy) CKRecord* storedCKRecord;

- (CKRecord*)CKRecordWithZoneID:(CKRecordZoneID*)zoneID;

// All of the following are virtual: you must override to use
- (NSString*)CKRecordName;
- (CKRecord*)updateCKRecord:(CKRecord*)record zoneID:(CKRecordZoneID*)zoneID;
- (void)setFromCKRecord:(CKRecord*)record;  // When you override this, make sure to call [setStoredCKRecord]
- (bool)matchesCKRecord:(CKRecord*)record;

@end

NS_ASSUME_NONNULL_END
#endif
