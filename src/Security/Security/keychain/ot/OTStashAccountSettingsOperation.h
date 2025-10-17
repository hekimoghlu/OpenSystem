/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ot/OctagonStateMachineHelpers.h"

@class OTAccountSettings;
@class OTOperationDependencies;
@class CuttlefishXPCWrapper;
@class TPSpecificUser;

NS_ASSUME_NONNULL_BEGIN

@protocol OTAccountSettingsContainer
- (void)setAccountSettings:(OTAccountSettings*_Nullable)accountSettings;
@end

@interface OTStashAccountSettingsOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                     accountSettings:(id<OTAccountSettingsContainer>)accountSettings
                         accountWide:(bool)accountWide
                          forceFetch:(bool)forceFetch;

+ (void)performWithAccountWide:(bool)accountWide
                    forceFetch:(bool)forceFetch
          cuttlefishXPCWrapper:(CuttlefishXPCWrapper*)cuttlefishXPCWrapper
                 activeAccount:(TPSpecificUser* _Nullable)activeAccount
                 containerName:(NSString*)containerName
                     contextID:(NSString*)contextID
                         reply:(void (^)(OTAccountSettings* _Nullable settings, NSError* _Nullable error))reply;

@end

NS_ASSUME_NONNULL_END

#endif
