/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#import <dispatch/dispatch.h>

#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ckks/CKKSKeychainBackedKey.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ckks/CKKSTLKShare.h"
#import "keychain/ckks/CKKSKeychainView.h"

NS_ASSUME_NONNULL_BEGIN

@interface OTFetchCKKSKeysOperation : CKKSGroupOperation

@property NSArray<CKKSKeychainBackedKeySet*>* viewKeySets;

// This contains all key sets which couldn't be converted to CKKSKeychainBackedKeySet, due to some error
@property NSArray<CKKSCurrentKeySet*>* incompleteKeySets;

// Any new TLKShares that CKKS suggested we upload along with this keyset
@property NSArray<CKKSTLKShare*>* pendingTLKShares;

// Any views that didn't provide a keyset within time
@property NSSet<CKRecordZoneID*>* zonesTimedOutWithoutKeysets;

// Set this to configure how long to wait for CKKS to resonse
@property dispatch_time_t desiredTimeout;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       refetchNeeded:(BOOL)refetchNeeded;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                        viewsToFetch:(NSSet<CKKSKeychainViewState*>*)views;

@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
