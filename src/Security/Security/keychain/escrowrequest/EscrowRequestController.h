/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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

extern NSString* const EscrowRequestTransitionErrorDomain;

extern OctagonState* const EscrowRequestStateNothingToDo;
extern OctagonState* const EscrowRequestStateTriggerCloudServices;
extern OctagonState* const EscrowRequestStateAttemptEscrowUpload;
extern OctagonState* const EscrowRequestStateWaitForUnlock;

@class CKKSLockStateTracker;

@interface EscrowRequestController : NSObject <OctagonStateMachineEngine>
@property OctagonStateMachine* stateMachine;

// Use this for testing: if set to true, we will always attempt to trigger CloudServices if needed, even if we've done it recently
@property bool forceIgnoreCloudServicesRateLimiting;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLockStateTracker:(CKKSLockStateTracker*)lockStateTracker;

- (void)triggerEscrowUpdateRPC:(nonnull NSString *)reason
                       options:(NSDictionary *)options
                         reply:(nonnull void (^)(NSError * _Nullable))reply;

- (void)storePrerecordsInEscrowRPC:(void (^)(uint64_t count, NSError* _Nullable error))reply;
@end

NS_ASSUME_NONNULL_END
