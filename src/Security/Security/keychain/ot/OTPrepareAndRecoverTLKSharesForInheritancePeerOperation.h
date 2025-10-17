/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#import <Foundation/Foundation.h>

#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ot/OctagonStateMachineHelpers.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ot/OTCuttlefishAccountStateHolder.h"
#import "keychain/ot/OTDeviceInformation.h"

NS_ASSUME_NONNULL_BEGIN


@interface OTPrepareAndRecoverTLKSharesForInheritancePeerOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>
@property OctagonState* nextState;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                                  ik:(OTInheritanceKey*)ik
                          deviceInfo:(OTDeviceInformation*)deviceInfo
                      policyOverride:(TPPolicyVersion* _Nullable)policyOverride
                  isInheritedAccount:(BOOL)isInheritedAccount
                               epoch:(uint64_t)epoch;

@property (nonatomic) uint64_t epoch;
@property OTDeviceInformation* deviceInfo;

@property (nullable) NSString* peerID;
@property (nullable) NSData* permanentInfo;
@property (nullable) NSData* permanentInfoSig;
@property (nullable) NSData* stableInfo;
@property (nullable) NSData* stableInfoSig;

@property (nullable) TPPolicyVersion* policyOverride;

@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
