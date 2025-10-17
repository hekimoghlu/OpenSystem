/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#import "keychain/ot/CuttlefishXPCWrapper.h"
#import "keychain/ot/OTOperationDependencies.h"

#import "keychain/ot/OTAuthKitAdapter.h"
#import "keychain/ot/OTConstants.h"
NS_ASSUME_NONNULL_BEGIN

@interface OTResetOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>

- (instancetype)init:(NSString*)containerName
           contextID:(NSString*)contextID
              reason:(CuttlefishResetReason)reason
   idmsTargetContext:(NSString *_Nullable)idmsTargetContext
idmsCuttlefishPassword:(NSString *_Nullable)idmsCuttlefishPassword
          notifyIdMS:(bool)notifyIdMS
       intendedState:(OctagonState*)intendedState
        dependencies:(OTOperationDependencies *)deps
          errorState:(OctagonState*)errorState
cuttlefishXPCWrapper:(CuttlefishXPCWrapper*)cuttlefishXPCWrapper;

@property CuttlefishResetReason resetReason;
@property (nullable) NSString* idmsTargetContext;
@property (nullable) NSString* idmsCuttlefishPassword;
@property () bool notifyIdMS;
@end

NS_ASSUME_NONNULL_END

#endif
