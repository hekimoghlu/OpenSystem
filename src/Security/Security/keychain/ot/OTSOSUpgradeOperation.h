/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#import <TrustedPeers/TrustedPeers.h>
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ot/OctagonStateMachineHelpers.h"
#import "keychain/ot/OTStates.h"
#import "keychain/ot/OTSOSAdapter.h"

#import "keychain/ot/proto/generated_source/OTAccountMetadataClassC.h"
#import "keychain/ot/OTDeviceInformation.h"

NS_ASSUME_NONNULL_BEGIN

@class OTOperationDependencies;

@interface OTSOSUpgradeOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>

@property (readonly, nullable) TPPolicyVersion* policyOverride;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                   ckksConflictState:(OctagonState*)ckksConflictState
                          errorState:(OctagonState*)errorState
                          deviceInfo:(OTDeviceInformation*)deviceInfo
                      policyOverride:(TPPolicyVersion* _Nullable)policyOverride;

@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON

