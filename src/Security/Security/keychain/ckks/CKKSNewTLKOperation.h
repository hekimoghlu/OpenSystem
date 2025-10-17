/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#import "keychain/ckks/CKKSProvideKeySetOperation.h"
#import "keychain/ckks/CKKSOperationDependencies.h"
#import "keychain/ot/OctagonStateMachine.h"

NS_ASSUME_NONNULL_BEGIN

@class CKKSKeychainView;

// This class will create+return the current key hierchies for all views

@interface CKKSNewTLKOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>
@property (readonly) CKKSOperationDependencies* deps;

@property (readonly, nullable) NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>* keysets;

- (instancetype)init NS_UNAVAILABLE;

// Any non-pending keysets provided to preexistingPendingKeySets will be ignored
- (instancetype)initWithDependencies:(CKKSOperationDependencies*)dependencies
                    rollTLKIfPresent:(BOOL)rollTLKIfPresent
           preexistingPendingKeySets:(NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>* _Nullable)previousPendingKeySets
                       intendedState:(CKKSState *)intendedState
                          errorState:(CKKSState *)errorState;

@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
