/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

#include <AssertMacros.h>

#import <Foundation/Foundation.h>
#import <Foundation/NSKeyedArchiver_Private.h>
#import "CKKSItem.h"
#import "CKKSSIV.h"

#include <utilities/SecDb.h>
#include "keychain/securityd/SecDbItem.h"
#include "keychain/securityd/SecItemSchema.h"

#import <CloudKit/CloudKit.h>

@implementation CKKSCKRecordHolder
@synthesize encodedCKRecord = _encodedCKRecord;
@synthesize storedCKRecord = _storedCKRecord;

- (instancetype)initWithCKRecord:(CKRecord*)record
                       contextID:(NSString*)contextID
{
    if(self = [super init]) {
        _zoneID = record.recordID.zoneID;
        _contextID = contextID;
        [self setFromCKRecord:record];
    }
    return self;
}

- (instancetype)initWithCKRecordType:(NSString*)recordType
                     encodedCKRecord:(NSData*)encodedCKRecord
                           contextID:(NSString*)contextID
                              zoneID:(CKRecordZoneID*)zoneID
{
    if(self = [super init]) {
        _contextID = contextID;
        _zoneID = zoneID;
        _ckRecordType = recordType;
        _encodedCKRecord = encodedCKRecord;
        _storedCKRecord = nil;
    }
    return self;
}

- (CKRecord*) storedCKRecord {
    if(_storedCKRecord != nil) {
        return [_storedCKRecord copy];
    }

    if(_encodedCKRecord == nil) {
        return nil;
    }
    @autoreleasepool {
        NSKeyedUnarchiver *coder = [[NSKeyedUnarchiver alloc] initForReadingFromData:_encodedCKRecord error:nil];
        CKRecord* ckRecord = [[CKRecord alloc] initWithCoder:coder];
        [coder finishDecoding];

        if(ckRecord && ![ckRecord.recordID.zoneID isEqual:self.zoneID]) {
            ckkserror("ckks", self.zoneID, "mismatching zone ids in a single record: %@ and %@", self.zoneID, ckRecord.recordID.zoneID);
        }

        _storedCKRecord = ckRecord;
        return [ckRecord copy];
    }
}

- (void) setStoredCKRecord: (CKRecord*) ckRecord {
    if(!ckRecord) {
        _encodedCKRecord = nil;
        _storedCKRecord = nil;
        return;
    }

    self.zoneID = ckRecord.recordID.zoneID;
    self.ckRecordType = ckRecord.recordType;

    @autoreleasepool {
        NSKeyedArchiver *archiver = [[NSKeyedArchiver alloc] initRequiringSecureCoding:YES];
        [ckRecord encodeWithCoder:archiver];
        _encodedCKRecord = archiver.encodedData;
        _storedCKRecord = [ckRecord copy];
    }
}

- (NSData*)encodedCKRecord
{
    return _encodedCKRecord;
}

- (void)setEncodedCKRecord:(NSData *)encodedCKRecord
{
    _encodedCKRecord = encodedCKRecord;
    _storedCKRecord = nil;
}

- (CKRecord*) CKRecordWithZoneID: (CKRecordZoneID*) zoneID {
    CKRecordID* recordID = [[CKRecordID alloc] initWithRecordName: [self CKRecordName] zoneID: zoneID];
    CKRecord* record = nil;

    if(self.encodedCKRecord == nil) {
        record = [[CKRecord alloc] initWithRecordType:self.ckRecordType recordID:recordID];
    } else {
        record = self.storedCKRecord;
    }

    CKRecord* originalRecord = [record copy];

    [self updateCKRecord:record zoneID:zoneID];

    if(![record isEqual:originalRecord]) {
        self.storedCKRecord = record;
    }
    return record;
}

- (NSString*) CKRecordName {
    @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                   reason:[NSString stringWithFormat:@"%@ must override %@", NSStringFromClass([self class]), NSStringFromSelector(_cmd)]
                                 userInfo:nil];
}
- (CKRecord*) updateCKRecord: (CKRecord*) record zoneID: (CKRecordZoneID*) zoneID {
    @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                   reason:[NSString stringWithFormat:@"%@ must override %@", NSStringFromClass([self class]), NSStringFromSelector(_cmd)]
                                 userInfo:nil];
}
- (void) setFromCKRecord: (CKRecord*) record {
    @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                   reason:[NSString stringWithFormat:@"%@ must override %@", NSStringFromClass([self class]), NSStringFromSelector(_cmd)]
                                 userInfo:nil];
}
- (bool) matchesCKRecord: (CKRecord*) record {
    @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                   reason:[NSString stringWithFormat:@"%@ must override %@", NSStringFromClass([self class]), NSStringFromSelector(_cmd)]
                                 userInfo:nil];
}

- (instancetype)copyWithZone:(NSZone *)zone {
    CKKSCKRecordHolder *rhCopy = [super copyWithZone:zone];
    rhCopy->_contextID = _contextID;
    rhCopy->_zoneID = _zoneID;
    rhCopy->_ckRecordType = _ckRecordType;
    rhCopy->_encodedCKRecord = [_encodedCKRecord copy];
    rhCopy->_storedCKRecord = [_storedCKRecord copy];
    return rhCopy;
}
@end

#endif
