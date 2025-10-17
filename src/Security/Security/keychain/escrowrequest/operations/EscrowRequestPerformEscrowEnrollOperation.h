/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#import <CoreCDP/CDPStateController.h>

#import "keychain/escrowrequest/generated_source/SecEscrowPendingRecord.h"
#import "keychain/ot/OctagonStateMachine.h"
#import "keychain/ckks/CKKSGroupOperation.h"

NS_ASSUME_NONNULL_BEGIN

@interface EscrowRequestPerformEscrowEnrollOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>

@property uint64_t numberOfRecordsUploaded;

- (instancetype)initWithIntendedState:(OctagonState*)intendedState
                           errorState:(OctagonState*)errorState
                  enforceRateLimiting:(bool)enforceRateLimiting
                     lockStateTracker:(CKKSLockStateTracker*)lockStateTracker;

+ (void)cdpUploadPrerecord:(SecEscrowPendingRecord*)recordToSend
                secretType:(CDPDeviceSecretType)secretType
                     reply:(void (^)(BOOL didUpdate, NSError* _Nullable error))reply;
@end

NS_ASSUME_NONNULL_END
