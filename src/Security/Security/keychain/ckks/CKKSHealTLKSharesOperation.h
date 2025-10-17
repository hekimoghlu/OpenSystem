/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#import "keychain/ckks/CKKSCurrentKeyPointer.h"
#import "keychain/ot/OctagonStateMachine.h"

#if OCTAGON

NS_ASSUME_NONNULL_BEGIN

@class CKKSKeychainView;
@class CKKSOperationDependencies;

@interface CKKSHealTLKSharesOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithDependencies:(CKKSOperationDependencies*)operationDependencies
                       intendedState:(CKKSState*)intendedState
                          errorState:(CKKSState*)errorState;


// For this keyset, who doesn't yet have a CKKSTLKShare for its TLK, shared to their current Octagon keys?
// Note that we really want a record sharing the TLK to ourselves, so this function might return
// a non-empty set even if all peers have the TLK: it wants us to make a record for ourself.
// If you pass in shares in keyset.pendingTLKShares, those records will be included in the calculation.
// If you do not pass a databaseProvider, this function will assume that you're already in a transaction.
+ (NSSet<CKKSTLKShareRecord*>* _Nullable)createMissingKeyShares:(CKKSCurrentKeySet*)keyset
                                                    trustStates:(NSArray<CKKSPeerProviderState*>*)trustStates
                                               databaseProvider:(id<CKKSDatabaseProviderProtocol> _Nullable)databaseProvider
                                                          error:(NSError* __autoreleasing*)errore;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
