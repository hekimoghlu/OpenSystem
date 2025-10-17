/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSRecordHolder.h"
#import "keychain/ckks/CKKSSQLDatabaseObject.h"

NS_ASSUME_NONNULL_BEGIN

@class CKKSWrappedAESSIVKey;

// Helper base class that includes UUIDs and key information
@interface CKKSItem : CKKSCKRecordHolder

@property (copy) NSString* uuid;
@property (copy) NSString* parentKeyUUID;
@property (nullable, copy) NSData* encitem;

@property (nullable, getter=base64Item, setter=setBase64Item:) NSString* base64encitem;

@property (nullable, copy) CKKSWrappedAESSIVKey* wrappedkey;
@property NSUInteger generationCount;
@property enum SecCKKSItemEncryptionVersion encver;

@property (nullable) NSNumber* plaintextPCSServiceIdentifier;
@property (nullable) NSData* plaintextPCSPublicKey;
@property (nullable) NSData* plaintextPCSPublicIdentity;

// Used for item encryption and decryption. Attempts to be future-compatible for new CloudKit record fields with an optional
// olditem field, which may contain a CK record. Any fields in that record that we don't understand will be added to the authenticated data dictionary.
- (NSDictionary<NSString*, NSData*>*)makeAuthenticatedDataDictionaryUpdatingCKKSItem:(CKKSItem* _Nullable)olditem
                                                                   encryptionVersion:(SecCKKSItemEncryptionVersion)encversion;


- (instancetype)initWithCKRecord:(CKRecord*)record
                       contextID:(NSString*)contextID;
- (instancetype)initCopyingCKKSItem:(CKKSItem*)item;

// Use this one if you really don't have any more information
- (instancetype)initWithUUID:(NSString*)uuid
               parentKeyUUID:(NSString*)parentKeyUUID
                   contextID:(NSString*)contextID
                      zoneID:(CKRecordZoneID*)zoneID;

// Use this one if you don't have a CKRecord yet
- (instancetype)initWithUUID:(NSString*)uuid
               parentKeyUUID:(NSString*)parentKeyUUID
                   contextID:(NSString*)contextID
                      zoneID:(CKRecordZoneID*)zoneID
                     encItem:(NSData* _Nullable)encitem
                  wrappedkey:(CKKSWrappedAESSIVKey* _Nullable)wrappedkey
             generationCount:(NSUInteger)genCount
                      encver:(NSUInteger)encver;

- (instancetype)initWithUUID:(NSString*)uuid
               parentKeyUUID:(NSString*)parentKeyUUID
                   contextID:(NSString*)contextID
                      zoneID:(CKRecordZoneID*)zoneID
             encodedCKRecord:(NSData* _Nullable)encodedrecord
                     encItem:(NSData* _Nullable)encitem
                  wrappedkey:(CKKSWrappedAESSIVKey* _Nullable)wrappedkey
             generationCount:(NSUInteger)genCount
                      encver:(NSUInteger)encver;

- (instancetype)initWithUUID:(NSString*)uuid
                    parentKeyUUID:(NSString*)parentKeyUUID
                        contextID:(NSString*)contextID
                           zoneID:(CKRecordZoneID*)zoneID
                  encodedCKRecord:(NSData* _Nullable)encodedrecord
                          encItem:(NSData* _Nullable)encitem
                       wrappedkey:(CKKSWrappedAESSIVKey* _Nullable)wrappedkey
                  generationCount:(NSUInteger)genCount
                           encver:(NSUInteger)encver
    plaintextPCSServiceIdentifier:(NSNumber* _Nullable)pcsServiceIdentifier
            plaintextPCSPublicKey:(NSData* _Nullable)pcsPublicKey
       plaintextPCSPublicIdentity:(NSData* _Nullable)pcsPublicIdentity;

// Convenience function: set the upload version for this record to be the current OS version
+ (void)setOSVersionInRecord:(CKRecord*)record;

+ (BOOL)intransactionRecordChanged:(CKRecord*)record
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;
+ (BOOL)intransactionRecordDeleted:(CKRecordID*)recordID
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;

@end

@interface CKKSSQLDatabaseObject (CKKSZoneExtras)
// Convenience function: get all UUIDs of this type on this particular zone
+ (NSArray<NSString*>*)allUUIDsWithContextID:(NSString*)contextID
                                      zoneID:(CKRecordZoneID*)zoneID
                                       error:(NSError * __autoreleasing *)error;

// Same as above, but allow for multiple zones at once
+ (NSSet<NSString*>*)allUUIDsWithContextID:(NSString*)contextID
                                   inZones:(NSSet<CKRecordZoneID*>*)zoneIDs
                                     error:(NSError * __autoreleasing *)error;

// Get all parentKeyUUIDs of this type in this particular zone
+ (NSSet<NSString*>*)allParentKeyUUIDsInContextID:(NSString*)contextID
                                           zoneID:(CKRecordZoneID*)zoneID
                                            error:(NSError * __autoreleasing *)error;

// Convenience function: get all objects in this particular zone
+ (NSArray*)allWithContextID:(NSString*)contextID
                      zoneID:(CKRecordZoneID*)zoneID
                       error:(NSError* _Nullable __autoreleasing* _Nullable)error;

// Convenience function: delete all records of this type with this zoneID
+ (bool)deleteAllWithContextID:(NSString*)contextID
                        zoneID:(CKRecordZoneID*)zoneID
                         error:(NSError* _Nullable __autoreleasing* _Nullable)error;
@end

NS_ASSUME_NONNULL_END
#endif
