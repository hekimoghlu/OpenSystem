/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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


#ifndef CKKSTests_MultiZone_h
#define CKKSTests_MultiZone_h

#if OCTAGON

#import "keychain/ckks/tests/CloudKitMockXCTest.h"
#import "keychain/ckks/tests/CloudKitKeychainSyncingMockXCTest.h"
#import "keychain/ckks/CKKSPeer.h"

@interface CloudKitKeychainSyncingMultiZoneTestsBase : CloudKitKeychainSyncingMockXCTest

@property CKRecordZoneID*      engramZoneID;
@property CKKSKeychainViewState* engramView;
@property FakeCKZone*          engramZone;
@property (readonly) ZoneKeys* engramZoneKeys;

@property CKRecordZoneID*      manateeZoneID;
@property CKKSKeychainViewState* manateeView;
@property FakeCKZone*          manateeZone;
@property (readonly) ZoneKeys* manateeZoneKeys;

@property CKRecordZoneID*      autoUnlockZoneID;
@property CKKSKeychainViewState* autoUnlockView;
@property FakeCKZone*          autoUnlockZone;
@property (readonly) ZoneKeys* autoUnlockZoneKeys;

@property CKRecordZoneID*      healthZoneID;
@property CKKSKeychainViewState* healthView;
@property FakeCKZone*          healthZone;
@property (readonly) ZoneKeys* healthZoneKeys;

@property CKRecordZoneID*      applepayZoneID;
@property CKKSKeychainViewState* applepayView;
@property FakeCKZone*          applepayZone;
@property (readonly) ZoneKeys* applepayZoneKeys;

@property CKRecordZoneID*      homeZoneID;
@property CKKSKeychainViewState* homeView;
@property FakeCKZone*          homeZone;
@property (readonly) ZoneKeys* homeZoneKeys;

@property CKRecordZoneID*      mfiZoneID;
@property CKKSKeychainViewState* mfiView;
@property FakeCKZone*          mfiZone;
@property (readonly) ZoneKeys* mfiZoneKeys;

@property CKRecordZoneID*      mailZoneID;
@property CKKSKeychainViewState* mailView;
@property FakeCKZone*          mailZone;
@property (readonly) ZoneKeys* mailZoneKeys;

@property CKRecordZoneID*      limitedZoneID;
@property CKKSKeychainViewState* limitedView;
@property FakeCKZone*          limitedZone;
@property (readonly) ZoneKeys* limitedZoneKeys;

@property CKRecordZoneID*      passwordsZoneID;
@property CKKSKeychainViewState* passwordsView;
@property FakeCKZone*          passwordsZone;
@property (readonly) ZoneKeys* passwordsZoneKeys;

@property CKRecordZoneID*      contactsZoneID;
@property CKKSKeychainViewState* contactsView;
@property FakeCKZone*          contactsZone;
@property (readonly) ZoneKeys* contactsZoneKeys;

@property CKRecordZoneID*      groupsZoneID;
@property CKKSKeychainViewState* groupsView;
@property FakeCKZone*          groupsZone;
@property (readonly) ZoneKeys* groupsZoneKeys;

@property CKRecordZoneID*      photosZoneID;
@property CKKSKeychainViewState* photosView;
@property FakeCKZone*          photosZone;
@property (readonly) ZoneKeys* photosZoneKeys;

@property CKRecordZoneID*      ptaZoneID;
@property CKKSKeychainViewState* ptaView;
@property FakeCKZone*          ptaZone;

- (void)saveFakeKeyHierarchiesToLocalDatabase;
- (void)putFakeDeviceStatusesInCloudKit;
- (void)putFakeKeyHierachiesInCloudKit;
- (void)saveTLKsToKeychain;
- (void)deleteTLKMaterialsFromKeychain;
- (void)waitForKeyHierarchyReadinesses;
- (void)expectCKKSTLKSelfShareUploads;

- (void)putAllFakeDeviceStatusesInCloudKit;
- (void)putAllSelfTLKSharesInCloudKit:(id<CKKSSelfPeer>)sharingPeer;
- (void)putAllTLKSharesInCloudKitFrom:(id<CKKSSelfPeer>)sharingPeer to:(id<CKKSPeer>)receivingPeer;

@end

#endif // OCTAGON
#endif /* CKKSTests_MultiZone_h */
