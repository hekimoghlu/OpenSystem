/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#import "keychain/ckks/CKKSGroupOperation.h"
#if OCTAGON

#import "keychain/ckks/CKKSOperationDependencies.h"
#import "keychain/ckks/CKKSMemoryKeyCache.h"
#import "keychain/ot/OctagonStateMachineHelpers.h"
#import "keychain/ot/OTPersonaAdapter.h"

NS_ASSUME_NONNULL_BEGIN

@class CKKSKeychainView;
@class CKKSItem;

@interface CKKSIncomingQueueOperation : CKKSResultOperation <OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;

// Set this to true if you're pretty sure that the policy set on the CKKS object
// should be considered authoritative, and items that do not match this policy should
// be moved.
@property bool handleMismatchedViewItems;

@property size_t successfulItemsProcessed;
@property size_t errorItemsProcessed;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithDependencies:(CKKSOperationDependencies*)dependencies
                           intending:(OctagonState*)intending
    pendingClassAItemsRemainingState:(OctagonState*)pendingClassAItemsState
                          errorState:(OctagonState*)errorState
           handleMismatchedViewItems:(bool)handleMismatchedViewItems;

// Use this to turn a CKKS item into a keychain dictionary suitable for keychain insertion
+ (NSDictionary* _Nullable)decryptCKKSItemToAttributes:(CKKSItem*)item
                                              keyCache:(CKKSMemoryKeyCache* _Nullable)keyCache
                           ckksOperationalDependencies:(CKKSOperationDependencies*)ckksOperationalDependencies
                                                 error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
