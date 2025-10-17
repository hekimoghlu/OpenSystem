/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
#ifndef CloudKitKeychainSyncingTestsBase_h
#define CloudKitKeychainSyncingTestsBase_h

#import <CloudKit/CloudKit.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wquoted-include-in-framework-header"
#import <OCMock/OCMock.h>
#pragma clang diagnostic pop

#import <XCTest/XCTest.h>

#include <Security/SecItemPriv.h>

#include "featureflags/affordance_featureflags.h"

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSKeychainViewState.h"
#import "keychain/ckks/CKKSViewManager.h"

#import "keychain/ckks/tests/CloudKitKeychainSyncingMockXCTest.h"
#import "keychain/ckks/tests/CloudKitMockXCTest.h"
#import "keychain/ckks/tests/MockCloudKit.h"

#import "keychain/ot/OTFollowup.h"
#import <CoreCDP/CDPFollowUpContext.h>
#import <CoreCDP/CDPAccount.h>

NS_ASSUME_NONNULL_BEGIN

@interface CloudKitKeychainSyncingTestsBase : CloudKitKeychainSyncingMockXCTest
@property (nullable) CKRecordZoneID* keychainZoneID;

@property (nullable) CKKSKeychainViewState* keychainView;

@property (nullable) FakeCKZone* keychainZone;
@property (nullable, readonly) ZoneKeys* keychainZoneKeys;

@property NSCalendar* utcCalendar;

- (ZoneKeys*)keychainZoneKeys;

@end

NS_ASSUME_NONNULL_END

#endif /* CloudKitKeychainSyncingTestsBase_h */
