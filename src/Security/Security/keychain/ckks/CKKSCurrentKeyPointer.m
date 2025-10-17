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
#import "CKKSCurrentKeyPointer.h"

#if OCTAGON

#import "keychain/ckks/CKKSStates.h"
#import "keychain/categories/NSError+UsefulConstructors.h"

@implementation CKKSCurrentKeyPointer

- (instancetype)initForClass:(CKKSKeyClass*)keyclass
                   contextID:(NSString*)contextID
              currentKeyUUID:(NSString*)currentKeyUUID
                      zoneID:(CKRecordZoneID*)zoneID
             encodedCKRecord: (NSData*) encodedrecord
{
    if(self = [super initWithCKRecordType: SecCKRecordCurrentKeyType
                          encodedCKRecord:encodedrecord
                                contextID:contextID
                                   zoneID:zoneID]) {
        _keyclass = keyclass;
        _currentKeyUUID = currentKeyUUID;

        if(self.currentKeyUUID == nil) {
            ckkserror_global("currentkey", "created a CKKSCurrentKey with a nil currentKeyUUID. Why?");
        }
    }
    return self;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSCurrentKeyPointer(%@, %@) %@: %@>",
            self.zoneID.zoneName,
            self.contextID,
            self.keyclass,
            self.currentKeyUUID];
}

- (instancetype)copyWithZone:(NSZone*)zone {
    CKKSCurrentKeyPointer* copy = [super copyWithZone:zone];
    copy.keyclass = [self.keyclass copyWithZone:zone];
    copy.currentKeyUUID = [self.currentKeyUUID copyWithZone:zone];
    return copy;
}
- (BOOL)isEqual: (id) object {
    if(![object isKindOfClass:[CKKSCurrentKeyPointer class]]) {
        return NO;
    }

    CKKSCurrentKeyPointer* obj = (CKKSCurrentKeyPointer*) object;

    return ([self.zoneID isEqual: obj.zoneID] &&
            ((self.currentKeyUUID == nil && obj.currentKeyUUID == nil) || [self.currentKeyUUID isEqual: obj.currentKeyUUID]) &&
            ((self.keyclass == nil && obj.keyclass == nil)             || [self.keyclass isEqual:obj.keyclass]) &&
            YES) ? YES : NO;
}

#pragma mark - CKKSCKRecordHolder methods

- (NSString*) CKRecordName {
    return self.keyclass;
}

- (CKRecord*) updateCKRecord: (CKRecord*) record zoneID: (CKRecordZoneID*) zoneID {
    if(![record.recordType isEqualToString: SecCKRecordCurrentKeyType]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordTypeException"
                reason:[NSString stringWithFormat: @"CKRecordType (%@) was not %@", record.recordType, SecCKRecordCurrentKeyType]
                userInfo:nil];
    }

    // The record name should already match keyclass...
    if(![record.recordID.recordName isEqualToString: self.keyclass]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordNameException"
                reason:[NSString stringWithFormat: @"CKRecord name (%@) was not %@", record.recordID.recordName, self.keyclass]
                userInfo:nil];
    }

    // Set the parent reference
    record[SecCKRecordParentKeyRefKey] = [[CKReference alloc] initWithRecordID: [[CKRecordID alloc] initWithRecordName: self.currentKeyUUID zoneID: zoneID] action: CKReferenceActionNone];
    return record;
}

- (bool) matchesCKRecord: (CKRecord*) record {
    if(![record.recordType isEqualToString: SecCKRecordCurrentKeyType]) {
        return false;
    }

    if(![record.recordID.recordName isEqualToString: self.keyclass]) {
        return false;
    }

    if(![[record[SecCKRecordParentKeyRefKey] recordID].recordName isEqualToString: self.currentKeyUUID]) {
        return false;
    }

    return true;
}

- (void) setFromCKRecord: (CKRecord*) record {
    if(![record.recordType isEqualToString: SecCKRecordCurrentKeyType]) {
        @throw [NSException
                exceptionWithName:@"WrongCKRecordTypeException"
                reason:[NSString stringWithFormat: @"CKRecordType (%@) was not %@", record.recordType, SecCKRecordCurrentKeyType]
                userInfo:nil];
    }

    [self setStoredCKRecord:record];

    // TODO: verify this is a real keyclass
    self.keyclass = (CKKSKeyClass*) record.recordID.recordName;
    self.currentKeyUUID = [record[SecCKRecordParentKeyRefKey] recordID].recordName;

    if(self.currentKeyUUID == nil) {
        ckkserror_global("currentkey", "No current key UUID in record! How/why? %@", record);
    }
}

#pragma mark - Load from database

+ (instancetype _Nullable)fromDatabase:(CKKSKeyClass*)keyclass
                             contextID:(NSString*)contextID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError * __autoreleasing *)error
{
    return [self fromDatabaseWhere: @{@"keyclass": keyclass,
                                      @"contextID": contextID,
                                      @"ckzone":zoneID.zoneName
                                    } error: error];
}

+ (instancetype _Nullable)tryFromDatabase:(CKKSKeyClass*)keyclass
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError * __autoreleasing *)error
{
    return [self tryFromDatabaseWhere: @{@"keyclass": keyclass,
                                         @"contextID": contextID,
                                         @"ckzone":zoneID.zoneName
                                       } error: error];
}

+ (instancetype _Nullable)forKeyClass:(CKKSKeyClass*)keyclass
                            contextID:(NSString*)contextID
                          withKeyUUID:(NSString*)keyUUID
                                zoneID:(CKRecordZoneID*)zoneID
                                error:(NSError * __autoreleasing *)error
{
    NSError* localerror = nil;
    CKKSCurrentKeyPointer* current = [self tryFromDatabase:keyclass
                                                 contextID:contextID
                                                    zoneID:zoneID
                                                     error:&localerror ];
    if(localerror) {
        if(error) {
            *error = localerror;
        }
        return nil;
    }

    if(current) {
        current.currentKeyUUID = keyUUID;
        return current;
    }

    return [[CKKSCurrentKeyPointer alloc] initForClass:keyclass
                                             contextID:contextID
                                        currentKeyUUID:keyUUID
                                                zoneID:zoneID
                                       encodedCKRecord:nil];
}

+ (NSArray<CKKSCurrentKeyPointer*>*)all:(CKRecordZoneID*)zoneID error: (NSError * __autoreleasing *) error {
    return [self allWhere:@{@"ckzone":zoneID.zoneName} error:error];
}

+ (bool) deleteAll:(CKRecordZoneID*) zoneID error: (NSError * __autoreleasing *) error {
    bool ok = [CKKSSQLDatabaseObject deleteFromTable:[self sqlTable] where: @{@"ckzone":zoneID.zoneName} connection:nil error: error];

    if(ok) {
        secdebug("ckksitem", "Deleted all %@", self);
    } else {
        secdebug("ckksitem", "Couldn't delete all %@: %@", self, error ? *error : @"unknown");
    }
    return ok;
}

#pragma mark - CKKSSQLDatabaseObject methods

+ (NSString*) sqlTable {
    return @"currentkeys";
}

+ (NSArray<NSString*>*) sqlColumns {
    return @[@"contextID", @"keyclass", @"currentKeyUUID", @"ckzone", @"ckrecord"];
}

- (NSDictionary<NSString*,NSString*>*) whereClauseToFindSelf {
    return @{@"contextID": self.contextID, @"keyclass": self.keyclass, @"ckzone":self.zoneID.zoneName};
}

- (NSDictionary<NSString*,NSString*>*) sqlValues {
    return @{@"keyclass": self.keyclass,
             @"contextID": self.contextID,
             @"currentKeyUUID": CKKSNilToNSNull(self.currentKeyUUID),
             @"ckzone":  CKKSNilToNSNull(self.zoneID.zoneName),
             @"ckrecord": CKKSNilToNSNull([self.encodedCKRecord base64EncodedStringWithOptions:0]),
             };
}

+ (instancetype)fromDatabaseRow:(NSDictionary<NSString*, CKKSSQLResult*>*)row {
    return [[CKKSCurrentKeyPointer alloc] initForClass:(CKKSKeyClass*)row[@"keyclass"].asString
                                             contextID:row[@"contextID"].asString
                                        currentKeyUUID:row[@"currentKeyUUID"].asString
                                                zoneID:[[CKRecordZoneID alloc] initWithZoneName:row[@"ckzone"].asString ownerName:CKCurrentUserDefaultName]
                                       encodedCKRecord:row[@"ckrecord"].asBase64DecodedData];
}

+ (BOOL)intransactionRecordChanged:(CKRecord*)record
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                       flagHandler:(id<OctagonStateFlagHandler> _Nullable)flagHandler
                             error:(NSError**)error
{
    // Pull out the old CKP, if it exists
    NSError* ckperror = nil;
    CKKSCurrentKeyPointer* oldckp = [CKKSCurrentKeyPointer tryFromDatabase:((CKKSKeyClass*)record.recordID.recordName)
                                                                 contextID:contextID
                                                                    zoneID:record.recordID.zoneID
                                                                     error:&ckperror];
    if(ckperror) {
        ckkserror("ckkskey", record.recordID.zoneID, "error loading ckp: %@", ckperror);
    }

    if(resync) {
        if(!oldckp) {
            ckkserror("ckksresync", record.recordID.zoneID, "BUG: No current key pointer matching resynced CloudKit record: %@", record);
        } else if(![oldckp matchesCKRecord:record]) {
            ckkserror("ckksresync", record.recordID.zoneID, "BUG: Local current key pointer doesn't match resynced CloudKit record: %@ %@", oldckp, record);
        } else {
            ckksnotice("ckksresync", record.recordID.zoneID, "Current key pointer has 'changed', but it matches our local copy: %@", record);
        }
    }

    NSError* localerror = nil;
    CKKSCurrentKeyPointer* currentkey = [[CKKSCurrentKeyPointer alloc] initWithCKRecord:record contextID:contextID];

    bool saved = [currentkey saveToDatabase:&localerror];
    if(!saved || localerror != nil) {
        ckkserror("ckkskey", record.recordID.zoneID, "Couldn't save current key pointer to database: %@: %@", currentkey, localerror);
        ckksinfo("ckkskey", record.recordID.zoneID, "CKRecord was %@", record);

        if(error) {
            *error = localerror;
        }
        return NO;
    }

    if([oldckp matchesCKRecord:record]) {
        ckksnotice("ckkskey", record.recordID.zoneID, "Current key pointer modification doesn't change anything interesting; skipping reprocess: %@", record);
    } else {
        // We've saved a new key in the database; trigger a rekey operation.
        [flagHandler _onqueueHandleFlag:CKKSFlagKeyStateProcessRequested];
    }

    return YES;
}

+ (BOOL)intransactionRecordDeleted:(CKRecordID*)recordID
                         contextID:(NSString*)contextID
                             error:(NSError**)error
{
    // Pull out the old CKP, if it exists
    NSError* ckperror = nil;
    CKKSCurrentKeyPointer* oldckp = [CKKSCurrentKeyPointer tryFromDatabase:((CKKSKeyClass*)recordID.recordName)
                                                                 contextID:contextID
                                                                    zoneID:recordID.zoneID
                                                                     error:&ckperror];
    if(ckperror) {
        ckkserror("ckkskey", recordID.zoneID, "error loading ckp: %@", ckperror);
        if(error) {
            *error = ckperror;
        }
        return NO;
    }

    if(!oldckp) {
        return YES;
    }

    NSError* deletionError = nil;
    [oldckp deleteFromDatabase:&deletionError];

    if(deletionError) {
        ckkserror("ckkskey", recordID.zoneID, "error deleting ckp: %@", deletionError);
        if(error) {
            *error = deletionError;
        }
        return NO;
    }

    return YES;
}

@end

@implementation CKKSCurrentKeySet
- (instancetype)initWithZoneID:(CKRecordZoneID*)zoneID
                     contextID:(NSString*)contextID
{
    if((self = [super init])) {
        _zoneID = zoneID;
        _contextID = contextID;
    }

    return self;
}

+ (CKKSCurrentKeySet*)loadForZone:(CKRecordZoneID*)zoneID
                        contextID:(NSString*)contextID
{
    @autoreleasepool {
        CKKSCurrentKeySet* set = [[CKKSCurrentKeySet alloc] initWithZoneID:zoneID contextID:contextID];
        NSError* error = nil;

        set.currentTLKPointer    = [CKKSCurrentKeyPointer tryFromDatabase: SecCKKSKeyClassTLK
                                                                contextID:contextID
                                                                   zoneID:zoneID
                                                                    error:&error];
        set.currentClassAPointer = [CKKSCurrentKeyPointer tryFromDatabase: SecCKKSKeyClassA
                                                                contextID:contextID
                                                                   zoneID:zoneID
                                                                    error:&error];
        set.currentClassCPointer = [CKKSCurrentKeyPointer tryFromDatabase: SecCKKSKeyClassC
                                                                contextID:contextID
                                                                   zoneID:zoneID
                                                                    error:&error];

        set.tlk    = set.currentTLKPointer.currentKeyUUID    ? [CKKSKey tryFromDatabase:set.currentTLKPointer.currentKeyUUID
                                                                              contextID:contextID
                                                                                 zoneID:zoneID
                                                                                  error:&error] : nil;
        set.classA = set.currentClassAPointer.currentKeyUUID ? [CKKSKey tryFromDatabase:set.currentClassAPointer.currentKeyUUID
                                                                              contextID:contextID
                                                                                 zoneID:zoneID
                                                                                  error:&error] : nil;
        set.classC = set.currentClassCPointer.currentKeyUUID ? [CKKSKey tryFromDatabase:set.currentClassCPointer.currentKeyUUID
                                                                              contextID:contextID
                                                                                 zoneID:zoneID
                                                                                  error:&error] : nil;

        set.pendingTLKShares = nil;

        set.proposed = NO;

        set.error = error;

        return set;
    }
}

-(NSString*)description {
    if(self.error) {
        return [NSString stringWithFormat:@"<CKKSCurrentKeySet[%@](%@): %@:%@ %@:%@ %@:%@ new:%d %@>",
                self.contextID,
                self.zoneID.zoneName,
                self.currentTLKPointer.currentKeyUUID, self.tlk,
                self.currentClassAPointer.currentKeyUUID, self.classA,
                self.currentClassCPointer.currentKeyUUID, self.classC,
                self.proposed,
                self.error];

    } else {
        return [NSString stringWithFormat:@"<CKKSCurrentKeySet[%@](%@): %@:%@ %@:%@ %@:%@ new:%d>",
                self.contextID,
                self.zoneID.zoneName,
                self.currentTLKPointer.currentKeyUUID, self.tlk,
                self.currentClassAPointer.currentKeyUUID, self.classA,
                self.currentClassCPointer.currentKeyUUID, self.classC,
                self.proposed];
    }
}
- (instancetype)copyWithZone:(NSZone*)zone {
    CKKSCurrentKeySet* copy = [[[self class] alloc] init];
    copy.zoneID = [self.zoneID copyWithZone:zone];
    copy.currentTLKPointer = [self.currentTLKPointer copyWithZone:zone];
    copy.currentClassAPointer = [self.currentClassAPointer copyWithZone:zone];
    copy.currentClassCPointer = [self.currentClassCPointer copyWithZone:zone];
    copy.tlk = [self.tlk copyWithZone:zone];
    copy.classA = [self.classA copyWithZone:zone];
    copy.classC = [self.classC copyWithZone:zone];
    copy.proposed = self.proposed;

    copy.error = [self.error copyWithZone:zone];
    return copy;
}

- (CKKSKeychainBackedKeySet* _Nullable)asKeychainBackedSet:(NSError**)error
{
    NSError* tlkError = nil;
    CKKSKeychainBackedKey* keychainBackedTLK = [self.tlk getKeychainBackedKey:&tlkError];

    NSError* classAError = nil;
    CKKSKeychainBackedKey* keychainBackedClassA = [self.classA getKeychainBackedKey:&classAError];

    NSError* classCError = nil;
    CKKSKeychainBackedKey* keychainBackedClassC = [self.classC getKeychainBackedKey:&classCError];

    if(keychainBackedTLK == nil ||
       keychainBackedClassA == nil ||
       keychainBackedClassC == nil) {
        if(error) {
            *error = [NSError errorWithDomain:CKKSErrorDomain
                                         code:CKKSKeysMissing
                                  description:@"unable to make keychain backed set; key is missing"
                                   underlying:tlkError ?: classAError ?: classCError];

        }
        return nil;
    }

    return [[CKKSKeychainBackedKeySet alloc] initWithTLK:keychainBackedTLK
                                                  classA:keychainBackedClassA
                                                  classC:keychainBackedClassC
                                               newUpload:self.proposed];
}
@end

#endif // OCTAGON
