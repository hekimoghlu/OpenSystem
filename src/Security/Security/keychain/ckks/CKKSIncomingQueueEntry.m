/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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

#import "CKKSKeychainView.h"

#include <utilities/SecDb.h>
#include "keychain/securityd/SecDbItem.h"
#include "keychain/securityd/SecItemSchema.h"

#import <CloudKit/CloudKit.h>
#import "CKKSIncomingQueueEntry.h"
#import "CKKSItemEncrypter.h"
#import "CKKSSIV.h"

@implementation CKKSIncomingQueueEntry

- (NSString*)description {
    return [NSString stringWithFormat: @"<%@[%@](%@): %@ %@ (%@)>",
            NSStringFromClass([self class]),
            self.item.contextID,
            self.item.zoneID.zoneName,
            self.action,
            self.item.uuid,
            self.state];
}

- (instancetype) initWithCKKSItem:(CKKSItem*) item
                           action:(NSString*) action
                            state:(NSString*) state {
    if(self = [super init]) {
        _item = item;
        _action = action;
        _state = state;
    }

    return self;
}

#pragma mark - Property access to underlying CKKSItem

- (NSString*)contextID
{
    return self.item.contextID;
}

-(NSString*)uuid {
    return self.item.uuid;
}

-(void)setUuid:(NSString *)uuid {
    self.item.uuid = uuid;
}

#pragma mark - Database Operations

+ (instancetype _Nullable)fromDatabase:(NSString*)uuid
                             contextID:(NSString*)contextID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError* __autoreleasing*)error{
    return [self fromDatabaseWhere: @{
        @"contextID": CKKSNilToNSNull(contextID),
        @"UUID": CKKSNilToNSNull(uuid),
        @"ckzone":CKKSNilToNSNull(zoneID.zoneName)
    } error: error];
}

+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                contextID:(NSString*)contextID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError* __autoreleasing*)error
{
    return [self tryFromDatabaseWhere: @{
        @"contextID": CKKSNilToNSNull(contextID),
        @"UUID": CKKSNilToNSNull(uuid),
        @"ckzone":CKKSNilToNSNull(zoneID.zoneName)
    } error: error];
}

+ (NSArray<CKKSIncomingQueueEntry*>*)fetch:(ssize_t)n
                            startingAtUUID:(NSString*)uuid
                                     state:(NSString*)state
                                    action:(NSString* _Nullable)action
                                 contextID:(NSString*)contextID
                                    zoneID:(CKRecordZoneID*)zoneID
                                     error: (NSError * __autoreleasing *) error {
    NSMutableDictionary* whereDict = [@{
        @"contextID": CKKSNilToNSNull(contextID),
        @"state": CKKSNilToNSNull(state),
        @"ckzone":CKKSNilToNSNull(zoneID.zoneName)
    } mutableCopy];
    whereDict[@"action"] = action;
    if(uuid) {
        whereDict[@"UUID"] = [CKKSSQLWhereValue op:CKKSSQLWhereComparatorGreaterThan value:uuid];
    }
    return [self fetch:n
                 where:whereDict
               orderBy:@[@"UUID"]
                 error:error];
}


#pragma mark - CKKSSQLDatabaseObject methods

+ (NSString*)sqlTable {
    return @"incomingqueue";
}

+ (NSArray<NSString*>*)sqlColumns {
    return [[CKKSItem sqlColumns] arrayByAddingObjectsFromArray: @[@"contextID", @"action", @"state"]];
}

- (NSDictionary<NSString*,NSString*>*)whereClauseToFindSelf {
    return @{@"contextID": self.contextID, @"UUID": self.uuid, @"state": self.state, @"ckzone": self.item.zoneID.zoneName};
}

- (NSDictionary<NSString*,NSString*>*)sqlValues {
    NSMutableDictionary* values = [[self.item sqlValues] mutableCopy];
    values[@"action"] = self.action;
    values[@"state"] = self.state;
    return values;
}


+ (instancetype)fromDatabaseRow:(NSDictionary<NSString *, CKKSSQLResult*>*) row {
    return [[CKKSIncomingQueueEntry alloc] initWithCKKSItem: [CKKSItem fromDatabaseRow: row]
                                                     action:row[@"action"].asString
                                                      state:row[@"state"].asString];
}

+ (NSDictionary<NSString*, NSNumber*>*)countsByStateWithContextID:(NSString*)contextID
                                                           zoneID:(CKRecordZoneID*)zoneID
                                                            error:(NSError* __autoreleasing*)error
{

    NSMutableDictionary* results = [[NSMutableDictionary alloc] init];

    [CKKSSQLDatabaseObject queryDatabaseTable: [[self class] sqlTable]
                                        where: @{
        @"contextID": CKKSNilToNSNull(contextID),
        @"ckzone": CKKSNilToNSNull(zoneID.zoneName)
    }
                                      columns: @[@"state", @"count(rowid)"]
                                      groupBy: @[@"state"]
                                      orderBy:nil
                                        limit: -1
                                   processRow: ^(NSDictionary<NSString*, CKKSSQLResult*>* row) {
                                       results[row[@"state"].asString] = row[@"count(rowid)"].asNSNumberInteger;
                                   }
                                        error: error];
    return results;
}

+ (NSInteger)countByState:(CKKSItemState *)state
                contextID:(NSString*)contextID
                     zone:(CKRecordZoneID*)zoneID
                    error:(NSError * __autoreleasing *)error
{
    __block NSInteger result = -1;

    [CKKSSQLDatabaseObject queryDatabaseTable: [[self class] sqlTable]
                                        where: @{
                                            @"contextID": CKKSNilToNSNull(contextID),
                                            @"ckzone": CKKSNilToNSNull(zoneID.zoneName),
                                            @"state": state
                                        }
                                        columns: @[@"count(*)"]
                                        groupBy: nil
                                        orderBy: nil
                                        limit: -1
                                        processRow: ^(NSDictionary<NSString*, CKKSSQLResult*>* row) {
                                            result = row[@"count(*)"].asNSInteger;
                                        }
                                        error: error];
    return result;
}

+ (NSDictionary<NSString*, NSNumber*>*)countNewEntriesByKeyWithContextID:(NSString*)contextID
                                                                  zoneID:(CKRecordZoneID*)zoneID
                                                                   error:(NSError* __autoreleasing*)error
{
    NSMutableDictionary* results = [[NSMutableDictionary alloc] init];

    [CKKSSQLDatabaseObject queryDatabaseTable:[[self class] sqlTable]
                                        where:@{
                                            @"contextID": CKKSNilToNSNull(contextID),
                                            @"ckzone": CKKSNilToNSNull(zoneID.zoneName),
                                            @"state": SecCKKSStateNew
                                        }
                                        columns:@[@"parentKeyUUID", @"count(rowid)"]
                                        groupBy:@[@"parentKeyUUID"]
                                        orderBy:nil
                                        limit:-1
                                        processRow:^(NSDictionary<NSString*, CKKSSQLResult*>* row) {
                                        results[row[@"parentKeyUUID"].asString] = row[@"count(rowid)"].asNSNumberInteger;
                                        }
                                        error: error];
    return results;
}


+ (BOOL)allIQEsHaveValidUnwrappingKeysInContextID:(NSString*)contextID
                                           zoneID:(CKRecordZoneID*)zoneID
                                            error:(NSError**)error
{
    NSError* parentKeyUUIDsError = nil;
    NSSet<NSString*>* parentKeyUUIDs = [CKKSIncomingQueueEntry allParentKeyUUIDsInContextID:contextID
                                                                                     zoneID:zoneID
                                                                           error:&parentKeyUUIDsError];

    if(parentKeyUUIDsError != nil) {
        ckkserror("ckkskey", zoneID, "Unable to find IQE parent keys: %@", parentKeyUUIDsError);
        if(error) {
            *error = parentKeyUUIDsError;
        }
        return NO;
    }

    for(NSString* parentKeyUUID in parentKeyUUIDs) {
        NSError* keyLoadError = nil;
        CKKSKey* parentKey = [CKKSKey tryFromDatabase:parentKeyUUID contextID:contextID zoneID:zoneID error:&keyLoadError];
        if(keyLoadError != nil) {
            // An error here means a database issue. Let's bail.
            ckkserror("ckksheal", zoneID, "Unable to find key %@: %@", keyLoadError, parentKeyUUID);
            if(error) {
                *error = keyLoadError;
            }
            return NO;
        } else if(parentKey == nil) {
            // No error but also no key means it doesn't exist. That's bad!
            ckksnotice("ckkskey", zoneID, "Some item is encrypted under a non-existent key(%@).", parentKeyUUID);
            return NO;
        }
    }

    // No issues found!
    return YES;
}

@end

#endif
