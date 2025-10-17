/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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


#import <Foundation/Foundation.h>
#import "keychain/ot/OctagonStateMachine.h"

NS_ASSUME_NONNULL_BEGIN

extern OctagonState* const KeychainItemUpgradeRequestStateNothingToDo;
extern OctagonState* const KeychainItemUpgradeRequestStateWaitForUnlock;
extern OctagonState* const KeychainItemUpgradeRequestStateUpgradePersistentRef;
extern OctagonFlag* const KeychainItemUpgradeRequestFlagSchedulePersistentReferenceUpgrade;
extern OctagonState* const KeychainItemUpgradeRequestStateWaitForTrigger;

@class CKKSLockStateTracker;
@class CKKSNearFutureScheduler;

@interface KeychainItemUpgradeRequestController : NSObject <OctagonStateMachineEngine>
@property OctagonStateMachine* stateMachine;
@property CKKSNearFutureScheduler* persistentReferenceUpgrader;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLockStateTracker:(CKKSLockStateTracker*)lockStateTracker;

- (void)triggerKeychainItemUpdateRPC:(nonnull void (^)(NSError * _Nullable))reply;

@end

NS_ASSUME_NONNULL_END
