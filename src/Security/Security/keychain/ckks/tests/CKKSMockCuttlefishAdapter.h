/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#ifndef CKKSMockCuttlefishAdapter_h
#define CKKSMockCuttlefishAdapter_h

#import <Foundation/Foundation.h>
#if OCTAGON

#import <CloudKit/CloudKit.h>
#import <CloudKit/CloudKit_Private.h>

#import "keychain/ckks/tests/CloudKitKeychainSyncingMockXCTest.h"
#import "keychain/ckks/tests/CloudKitMockXCTest.h"
#import "keychain/ckks/tests/MockCloudKit.h"

#import "keychain/ckks/CKKSCuttlefishAdapter.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSMockCuttlefishAdapter : NSObject <CKKSCuttlefishAdapterProtocol>

@property (nullable) NSMutableDictionary<CKRecordZoneID*, FakeCKZone*>* fakeCKZones;
@property (nullable) NSMutableDictionary<CKRecordZoneID*, ZoneKeys*>* zoneKeys;
@property (nullable) NSString* peerID;

- (instancetype)init:(NSMutableDictionary<CKRecordZoneID*, FakeCKZone*>*)fakeCKZones
            zoneKeys:(NSMutableDictionary<CKRecordZoneID*, ZoneKeys*>*)zoneKeys
              peerID:(NSString*)peerID;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON

#endif
