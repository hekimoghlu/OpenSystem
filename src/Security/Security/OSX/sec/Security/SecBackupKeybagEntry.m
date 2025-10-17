/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include <AssertMacros.h>

#import <Foundation/Foundation.h>

#include <utilities/SecDb.h>
#include "keychain/securityd/SecDbItem.h"
#include "keychain/securityd/SecItemSchema.h"

#if OCTAGON

#import "SecBackupKeybagEntry.h"

// from CKKSZoneStateEntry.m

@implementation SecBackupKeybagEntry

- (instancetype) initWithPublicKey: (NSData*)publicKey publickeyHash: (NSData*) publickeyHash user: (NSData*) user {
    if (self = [super init]) {
        _publickey = publicKey;
        _publickeyHash = publickeyHash;
        _musr = user;
    }
    return self;
}

- (BOOL)isEqual: (id) object {
    if(![object isKindOfClass:[SecBackupKeybagEntry class]]) {
        return NO;
    }

    SecBackupKeybagEntry* obj = (SecBackupKeybagEntry*) object;

    return ([self.publickeyHash isEqual: obj.publickeyHash]) ? YES : NO;
}

+ (instancetype) state: (NSData*) publickeyHash {
    NSError* error = nil;
    SecBackupKeybagEntry* ret = [SecBackupKeybagEntry tryFromDatabase:publickeyHash error:&error];

    if (error) {
        secerror("CKKS: error fetching SecBackupKeybagEntry(%@): %@", publickeyHash, error);
    }

    if(!ret) {
        ret = [[SecBackupKeybagEntry alloc] initWithPublicKey: nil publickeyHash: (NSData*) publickeyHash user: nil];
    }
    return ret;
}

#pragma mark - Database Operations

+ (instancetype) fromDatabase: (NSData*) publickeyHash error: (NSError * __autoreleasing *) error {
    return [self fromDatabaseWhere: @{@"publickeyHash": publickeyHash} error: error];
}

+ (instancetype) tryFromDatabase: (NSData*) publickeyHash error: (NSError * __autoreleasing *) error {
    return [self tryFromDatabaseWhere: @{@"publickeyHash": publickeyHash} error: error];
}

#pragma mark - CKKSSQLDatabaseObject methods

+ (NSString*) sqlTable {
    return @"backup_keybag";
}

+ (NSArray<NSString*>*) sqlColumns {
    return @[@"publickey", @"publickeyHash", @"musr"];
}

- (NSDictionary<NSString*,id>*) whereClauseToFindSelf {
    return @{@"publickeyHash": self.publickeyHash};
}

// used by saveToDatabaseWithConnection to write to db
- (NSDictionary<NSString*,id>*) sqlValues {
    return @{
        @"publickey":       [self.publickey base64EncodedStringWithOptions:0],
        @"publickeyHash":   [self.publickeyHash base64EncodedStringWithOptions:0],
        @"musr":            [self.musr base64EncodedStringWithOptions:0],
    };
}

+ (instancetype)fromDatabaseRow:(NSDictionary<NSString*, CKKSSQLResult*>*)row {
    NSData *publicKey = row[@"publickey"].asBase64DecodedData;
    NSData *publickeyHash = row[@"publickeyHash"].asBase64DecodedData;
    NSData *musr = row[@"musr"].asBase64DecodedData;
    if (publicKey == NULL || publickeyHash == NULL || musr == NULL) {
        return nil;
    }

    return [[SecBackupKeybagEntry alloc] initWithPublicKey:publicKey publickeyHash:publickeyHash user:musr];
}

@end

#endif
