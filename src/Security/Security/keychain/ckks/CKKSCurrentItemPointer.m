/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#import "keychain/ckks/CKKSCurrentItemPointer.h"

#if OCTAGON

@implementation CKKSCurrentItemPointer

- (instancetype)initForIdentifier:(NSString*)identifier
                        contextID:(NSString*)contextID
                  currentItemUUID:(NSString*)currentItemUUID
                            state:(CKKSProcessedState*)state
                           zoneID:(CKRecordZoneID*)zoneID
                  encodedCKRecord: (NSData*) encodedrecord
{
    if(self = [super initWithCKRecordType: SecCKRecordCurrentItemType
                          encodedCKRecord:encodedrecord
                                contextID:contextID
                                   zoneID:zoneID]) {
        _state = state;
        _identifier = identifier;
        _currentItemUUID = currentItemUUID;
    }
    return self;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSCurrentItemPointer[%@](%@) %@: %@>",
            self.contextID,
            self.zoneID.zoneName,
            self.identifier,
            self.currentItemUUID];
}

#pragma mark - CKKSCKRecordHolder methods

- (NSString*) CKRecordName {
    return self.identifier;
}

- (CKRecord*)updateCKRecord: (CKRecord*) record zoneID: (CKRecordZoneID*) zoneID {
    if(![record.recordType isEqualToString: SecCKRecordCurrentItemType]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordTypeException"
                reason:[NSString stringWithFormat: @"CKRecordType (%@) was not %@", record.recordType, SecCKRecordCurrentItemType]
                userInfo:nil];
    }

    // The record name should already match identifier...
    if(![record.recordID.recordName isEqualToString: self.identifier]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordNameException"
                reason:[NSString stringWithFormat: @"CKRecord name (%@) was not %@", record.recordID.recordName, self.identifier]
                userInfo:nil];
    }

    // Set the parent reference
    record[SecCKRecordItemRefKey] = [[CKReference alloc] initWithRecordID: [[CKRecordID alloc] initWithRecordName: self.currentItemUUID zoneID: zoneID]
                                                                   action: CKReferenceActionNone];
    return record;
}

- (bool)matchesCKRecord: (CKRecord*) record {
    if(![record.recordType isEqualToString: SecCKRecordCurrentItemType]) {
        return false;
    }

    if(![record.recordID.recordName isEqualToString: self.identifier]) {
        return false;
    }

    if(![[record[SecCKRecordItemRefKey] recordID].recordName isEqualToString: self.currentItemUUID]) {
        return false;
    }

    return true;
}

- (void)setFromCKRecord: (CKRecord*) record {
    if(![record.recordType isEqualToString: SecCKRecordCurrentItemType]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordTypeException"
                reason:[NSString stringWithFormat: @"CKRecordType (%@) was not %@", record.recordType, SecCKRecordCurrentItemType]
                userInfo:nil];
    }

    [self setStoredCKRecord:record];

    self.identifier = (CKKSKeyClass*) record.recordID.recordName;
    self.currentItemUUID = [record[SecCKRecordItemRefKey] recordID].recordName;
}

#pragma mark - Load from database

+ (instancetype)fromDatabase:(NSString*)identifier
                   contextID:(NSString*)contextID
                       state:(CKKSProcessedState*)state
                      zoneID:(CKRecordZoneID*)zoneID error: (NSError * __autoreleasing *) error {
    return [self fromDatabaseWhere: @{@"contextID":contextID, @"identifier":identifier, @"state":state, @"ckzone":zoneID.zoneName} error: error];
}

+ (instancetype)tryFromDatabase:(NSString*)identifier
                      contextID:(NSString*)contextID
                          state:(CKKSProcessedState*)state
                         zoneID:(CKRecordZoneID*)zoneID
                          error: (NSError * __autoreleasing *) error {
    return [self tryFromDatabaseWhere: @{@"contextID":contextID, @"identifier":identifier, @"state":state, @"ckzone":zoneID.zoneName} error: error];
}

+ (NSArray<CKKSCurrentItemPointer*>*)remoteItemPointers: (CKRecordZoneID*)zoneID
                                              contextID:(NSString*)contextID
                                                  error: (NSError * __autoreleasing *) error {
    return [self allWhere: @{@"state":  SecCKKSProcessedStateRemote, @"ckzone":zoneID.zoneName} error:error];
}

+ (NSArray<CKKSCurrentItemPointer*>*)allInZone:(CKRecordZoneID*)zoneID
                                     contextID:(NSString*)contextID
                                         error:(NSError * __autoreleasing *)error {
    return [self allWhere: @{@"ckzone":zoneID.zoneName} error:error];
}

+ (bool)deleteAll:(CKRecordZoneID*)zoneID
        contextID:(NSString*)contextID
            error:(NSError * __autoreleasing *)error {
    bool ok = [CKKSSQLDatabaseObject deleteFromTable:[self sqlTable] where: @{@"ckzone":zoneID.zoneName} connection:nil error: error];

    if(ok) {
        secdebug("ckksitem", "Deleted all %@", self);
    } else {
        secdebug("ckksitem", "Couldn't delete all %@: %@", self, error ? *error : @"unknown");
    }
    return ok;
}

#pragma mark - CKKSSQLDatabaseObject methods

+ (NSString*)sqlTable {
    return @"currentitems";
}

+ (NSArray<NSString*>*)sqlColumns {
    return @[@"contextID", @"identifier", @"currentItemUUID", @"state", @"ckzone", @"ckrecord"];
}

- (NSDictionary<NSString*,NSString*>*) whereClauseToFindSelf {
    return @{@"contextID": CKKSNilToNSNull(self.contextID),
             @"identifier": self.identifier,
             @"ckzone":self.zoneID.zoneName,
             @"state":self.state};
}

- (NSDictionary<NSString*,NSString*>*)sqlValues {
    return @{@"contextID": CKKSNilToNSNull(self.contextID),
             @"identifier": self.identifier,
             @"currentItemUUID": CKKSNilToNSNull(self.currentItemUUID),
             @"state": CKKSNilToNSNull(self.state),
             @"ckzone":  CKKSNilToNSNull(self.zoneID.zoneName),
             @"ckrecord": CKKSNilToNSNull([self.encodedCKRecord base64EncodedStringWithOptions:0]),
             };
}

+ (instancetype)fromDatabaseRow:(NSDictionary<NSString *, CKKSSQLResult*>*) row {
    return [[CKKSCurrentItemPointer alloc] initForIdentifier:row[@"identifier"].asString
                                                   contextID:row[@"contextID"].asString
                                             currentItemUUID:row[@"currentItemUUID"].asString
                                                       state:(CKKSProcessedState*)row[@"state"].asString
                                                      zoneID:[[CKRecordZoneID alloc] initWithZoneName:row[@"ckzone"].asString ownerName:CKCurrentUserDefaultName]
                                             encodedCKRecord:row[@"ckrecord"].asBase64DecodedData];
}

+ (NSInteger)countByState:(CKKSItemState *)state
                contextID:(NSString*)contextID
                     zone:(CKRecordZoneID*)zoneID
                    error:(NSError * __autoreleasing *)error
{
    __block NSInteger result = -1;

    [CKKSSQLDatabaseObject queryDatabaseTable:[[self class] sqlTable]
                                        where:@{@"contextID": CKKSNilToNSNull(contextID),
                                                @"ckzone": CKKSNilToNSNull(zoneID.zoneName),
                                                @"state": state }
                                      columns:@[@"count(*)"]
                                      groupBy:nil
                                      orderBy:nil
                                        limit:-1
                                   processRow:^(NSDictionary<NSString*, CKKSSQLResult*>* row) {
                                       result = row[@"count(*)"].asNSInteger;
                                   }
                                        error: error];
    return result;
}

+ (BOOL)intransactionRecordChanged:(CKRecord*)record
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error
{
    if(resync) {
        NSError* ciperror = nil;
        CKKSCurrentItemPointer* localcip  = [CKKSCurrentItemPointer tryFromDatabase:record.recordID.recordName
                                                                          contextID:contextID
                                                                              state:SecCKKSProcessedStateLocal
                                                                             zoneID:record.recordID.zoneID
                                                                              error:&ciperror];
        CKKSCurrentItemPointer* remotecip = [CKKSCurrentItemPointer tryFromDatabase:record.recordID.recordName
                                                                          contextID:contextID
                                                                              state:SecCKKSProcessedStateRemote
                                                                             zoneID:record.recordID.zoneID
                                                                              error:&ciperror];
        if(ciperror) {
            ckkserror("ckksresync", record.recordID.zoneID, "error loading cip: %@", ciperror);
        }
        if(!(localcip || remotecip)) {
            ckkserror("ckksresync", record.recordID.zoneID, "BUG: No current item pointer matching resynced CloudKit record: %@", record);
        } else if(! ([localcip matchesCKRecord:record] || [remotecip matchesCKRecord:record]) ) {
            ckkserror("ckksresync", record.recordID.zoneID, "BUG: Local current item pointer doesn't match resynced CloudKit record(s): %@ %@ %@", localcip, remotecip, record);
        } else {
            ckksnotice("ckksresync", record.recordID.zoneID, "Already know about this current item pointer, skipping update: %@", record);
            return YES;
        }
    }

    NSError* localerror = nil;
    CKKSCurrentItemPointer* cip = [[CKKSCurrentItemPointer alloc] initWithCKRecord:record contextID:contextID];
    cip.state = SecCKKSProcessedStateRemote;

    bool saved = [cip saveToDatabase:&localerror];
    if(!saved || localerror) {
        ckkserror("currentitem", record.recordID.zoneID, "Couldn't save current item pointer to database: %@: %@ %@", cip, localerror, record);
        if(error) {
            *error = localerror;
        }
        return NO;
    }
    return YES;
}

+ (BOOL)intransactionRecordDeleted:(CKRecordID*)recordID
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error
{
    NSError* localerror = nil;
    ckksinfo("currentitem", recordID.zoneID, "CloudKit notification: deleted current item pointer(%@): %@", SecCKRecordCurrentItemType, recordID);

    CKKSCurrentItemPointer* remote = [CKKSCurrentItemPointer tryFromDatabase:[recordID recordName]
                                                                   contextID:contextID
                                                                       state:SecCKKSProcessedStateRemote
                                                                      zoneID:recordID.zoneID
                                                                       error:&localerror];
    if(localerror) {
        if(error) {
            *error = localerror;
        }

        ckkserror("currentitem", recordID.zoneID, "Failed to find remote CKKSCurrentItemPointer to delete %@: %@", recordID, localerror);
        return NO;
    }

    [remote deleteFromDatabase:&localerror];
    if(localerror) {
        if(error) {
            *error = localerror;
        }
        ckkserror("currentitem", recordID.zoneID, "Failed to delete remote CKKSCurrentItemPointer %@: %@", recordID, localerror);
        return NO;
    }

    CKKSCurrentItemPointer* local = [CKKSCurrentItemPointer tryFromDatabase:[recordID recordName]
                                                                  contextID:contextID
                                                                      state:SecCKKSProcessedStateLocal
                                                                     zoneID:recordID.zoneID
                                                                      error:&localerror];
    if(localerror) {
        if(error) {
            *error = localerror;
        }
        ckkserror("currentitem", recordID.zoneID, "Failed to find local CKKSCurrentItemPointer %@: %@", recordID, localerror);
        return NO;
    }
    [local deleteFromDatabase:&localerror];
    if(localerror) {
        if(error) {
            *error = localerror;
        }
        ckkserror("currentitem", recordID.zoneID, "Failed to delete local CKKSCurrentItemPointer %@: %@", recordID, localerror);
        return NO;
    }

    ckksinfo("currentitem", recordID.zoneID, "CKKSCurrentItemPointer was deleted: %@ error: %@", recordID, localerror);

    if(error && localerror) {
        *error = localerror;
    }

    return (localerror == nil);
}

@end

#endif // OCTAGON

