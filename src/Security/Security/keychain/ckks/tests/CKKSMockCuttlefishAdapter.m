/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

#import <SecurityFoundation/SecurityFoundation.h>

#import <CloudKit/CloudKit.h>
#import <CloudKit/CloudKit_Private.h>

#import "keychain/ckks/CKKSCuttlefishAdapter.h"
#import "keychain/ckks/tests/CKKSMockCuttlefishAdapter.h"
#import "keychain/ckks/tests/MockCloudKit.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"

@implementation CKKSMockCuttlefishAdapter

- (instancetype)init:(NSMutableDictionary<CKRecordZoneID*, FakeCKZone*>*)fakeCKZones
            zoneKeys:(NSMutableDictionary<CKRecordZoneID*, ZoneKeys*>*)zoneKeys
              peerID:(NSString*)peerID {
    
    if((self = [super init])) {
        _fakeCKZones = fakeCKZones;
        _zoneKeys = zoneKeys;
        _peerID = peerID;
    }
    return self;
}

- (void)fetchCurrentItem:(TPSpecificUser* __nullable)activeAccount
                   items:(nonnull NSArray<CuttlefishCurrentItemSpecifier *> *)items
                   reply:(nonnull void (^)(NSArray<CuttlefishCurrentItem *> * _Nullable, NSArray<CKRecord *> * _Nullable, NSError * _Nullable))reply
{
    NSMutableArray<CuttlefishCurrentItem*>* cuttlefishCurrentItems = [NSMutableArray array];
    NSMutableSet<CKRecord*>* syncKeys = [NSMutableSet set];
    
    for (CuttlefishCurrentItemSpecifier* item in items) {
        
        // Grab zone
        FakeCKZone* zone = nil;
        for (CKRecordZoneID* zoneID in self.fakeCKZones) {
            if ([zoneID.zoneName isEqualToString:item.zoneID]) {
                zone = self.fakeCKZones[zoneID];
                break;
            }
        }
        
        if (zone == nil) {
            reply(nil, nil, [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchView userInfo:@{@"description": [NSString stringWithFormat:@"No such zone (%@)",item.zoneID]}]);
            return;
        }
        
        CKRecordZoneID* recordZoneID = [[CKRecordZoneID alloc] initWithZoneName:item.zoneID ownerName:CKCurrentUserDefaultName];
        
        CKRecordID* currentItemRecordID = [[CKRecordID alloc] initWithRecordName:item.itemPtrName zoneID:recordZoneID];
        CKRecord* currentItemRecord = zone.currentDatabase[currentItemRecordID];
        
        if (currentItemRecord == nil) {
            reply(nil, nil, [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No such currentitem record (%@)", item]}]);
            return;
        }
        
        CKRecordID* itemRecordID = [[CKRecordID alloc] initWithRecordName: [currentItemRecord[SecCKRecordItemRefKey] recordID].recordName zoneID:recordZoneID];
        CKRecord* itemRecord = zone.currentDatabase[itemRecordID];
        
        if (itemRecord == nil) {
            reply(nil, nil, [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No such item record for current item pointer (%@)", item]}]);
            return;
        }
        
        [cuttlefishCurrentItems addObject:[[CuttlefishCurrentItem alloc] init:item item:itemRecord]];
        
        // For now, assuming that no TLKs have been rolled.
        [syncKeys addObject:[self.zoneKeys[recordZoneID].tlk CKRecordWithZoneID:recordZoneID]];

        NSString* parentKeyUUID = [itemRecord[SecCKRecordParentKeyRefKey] recordID].recordName;
        if ([parentKeyUUID isEqualToString:self.zoneKeys[recordZoneID].classA.uuid]) {
            [syncKeys addObject:[self.zoneKeys[recordZoneID].classA CKRecordWithZoneID:recordZoneID]];
        } else if ([parentKeyUUID isEqualToString:self.zoneKeys[recordZoneID].classC.uuid]) {
            [syncKeys addObject:[self.zoneKeys[recordZoneID].classC CKRecordWithZoneID:recordZoneID]];
        } else {
            NSError* error = [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No parent record for %@", itemRecord.recordID.recordName]}];
            ckkserror_global("ckks-mockcuttlefish", "Identity wrapped by parent key that does not exist");
            reply(nil, nil, error);
            return;
        }

    }

    reply(cuttlefishCurrentItems, [syncKeys allObjects], nil);
    
}

- (void)fetchPCSIdentityByKey:(TPSpecificUser* __nullable)activeAccount
                  pcsservices:(nonnull NSArray<CuttlefishPCSServiceIdentifier *> *)pcsservices
                        reply:(nonnull void (^)(NSArray<CuttlefishPCSIdentity *> * _Nullable, NSArray<CKRecord *> * _Nullable, NSError * _Nullable))reply
{
    
    NSMutableArray<CuttlefishPCSIdentity*>* pcsIdentities = [NSMutableArray array];
    NSMutableSet<CKRecord*>* syncKeys = [NSMutableSet set];
    
    for (CuttlefishPCSServiceIdentifier* pcsService in pcsservices) {
        
        // Grab zone
        FakeCKZone* zone = nil;
        for (CKRecordZoneID* zoneID in self.fakeCKZones) {
            if ([zoneID.zoneName isEqualToString:pcsService.zoneID]) {
                zone = self.fakeCKZones[zoneID];
                break;
            }
        }
        
        if (zone == nil) {
            reply(nil, nil, [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchView userInfo:@{@"description": [NSString stringWithFormat:@"No such zone (%@)",pcsService.zoneID]}]);
            return;
        }
        
        // Find record with matching service number and public key
        CKRecord* pcsIdentityRecord = nil;
        for (CKRecord* record in [zone.currentDatabase allValues]) {
            if (([record.recordType isEqualToString:SecCKRecordItemRefKey])
                && ([record[SecCKRecordPCSServiceIdentifier] isEqual: pcsService.PCSServiceID])
                && ([record[SecCKRecordPCSPublicKey] isEqual: pcsService.PCSPublicKey])) {
                pcsIdentityRecord = record;
                break;
            }
        }
        
        if (pcsIdentityRecord == nil) {
            reply(nil, nil, [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No such item record (%@)", pcsService]}]);
            return;
        }
        
        // Return service + item
        [pcsIdentities addObject:[[CuttlefishPCSIdentity alloc] init:pcsService item:pcsIdentityRecord]];

        CKRecordZoneID* recordZoneID = [[CKRecordZoneID alloc] initWithZoneName:pcsService.zoneID ownerName:CKCurrentUserDefaultName];

        // For now, assuming that no TLKs have been rolled.
        [syncKeys addObject:[self.zoneKeys[recordZoneID].tlk CKRecordWithZoneID:recordZoneID]];

        NSString* parentKeyUUID = [pcsIdentityRecord[SecCKRecordParentKeyRefKey] recordID].recordName;
        if ([parentKeyUUID isEqualToString:self.zoneKeys[recordZoneID].classA.uuid]) {
            [syncKeys addObject:[self.zoneKeys[recordZoneID].classA CKRecordWithZoneID:recordZoneID]];
        } else if ([parentKeyUUID isEqualToString:self.zoneKeys[recordZoneID].classC.uuid]) {
            [syncKeys addObject:[self.zoneKeys[recordZoneID].classC CKRecordWithZoneID:recordZoneID]];
        } else {
            NSError* error = [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No parent record for %@", pcsIdentityRecord.recordID.recordName]}];
            ckkserror_global("ckks-mockcuttlefish", "Identity wrapped by parent key that does not exist");
            reply(nil, nil, error);
            return;
        }
    }

    reply(pcsIdentities, [syncKeys allObjects], nil);
}

- (void)fetchRecoverableTLKShares:(TPSpecificUser *)activeAccount 
                           peerID:(NSString *)peerID
                        contextID:(NSString *)contextID
                          altDSID:(NSString * _Nullable)altDSID
                           flowID:(NSString * _Nullable)flowID
                  deviceSessionID:(NSString * _Nullable)deviceSessionID
                   canSendMetrics:(BOOL)canSendMetrics
                            reply:(void (^)(NSArray<CKKSTLKShareRecord *> * _Nullable, NSError * _Nullable))reply {
    NSMutableArray<CKKSTLKShareRecord*>* tlkShareRecords = [[NSMutableArray alloc] init];
    for (CKRecordZoneID* zoneID in self.fakeCKZones) {
        FakeCKZone* zone = self.fakeCKZones[zoneID];
        for (CKRecord* record in [zone.currentDatabase allValues]) {
            if (([record.recordType isEqualToString:SecCKRecordTLKShareType])) {
                CKKSTLKShareRecord* tlkShareRecord = [[CKKSTLKShareRecord alloc] initWithCKRecord:record contextID:contextID];
                if ([tlkShareRecord.share.receiverPeerID isEqualToString:self.peerID]) {
                    [tlkShareRecords addObject:tlkShareRecord];
                }
            }
        }
    }

    NSError* error = nil;
    if (tlkShareRecords.count == 0) {
        error = [NSError errorWithDomain:CKKSErrorDomain code:CKKSNoSuchRecord userInfo:@{@"description": [NSString stringWithFormat:@"No TLK Shares for peer (%@)", self.peerID]}];
        tlkShareRecords = nil;
    }

    reply(tlkShareRecords, error);
}

@end
