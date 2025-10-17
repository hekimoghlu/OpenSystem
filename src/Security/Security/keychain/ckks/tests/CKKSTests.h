/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#import <OCMock/OCMock.h>
#import <XCTest/XCTest.h>

#include <Security/SecItemPriv.h>
#include "OSX/sec/Security/SecItemShim.h"

#import "keychain/ckks/tests/CloudKitKeychainSyncingTestsBase.h"

NS_ASSUME_NONNULL_BEGIN

// 3 keys, 3 current keys, and 1 device state entry
#define SYSTEM_DB_RECORD_COUNT (7)

@interface CloudKitKeychainSyncingTests : CloudKitKeychainSyncingTestsBase
@end

NS_ASSUME_NONNULL_END

#endif /* OCTAGON */
