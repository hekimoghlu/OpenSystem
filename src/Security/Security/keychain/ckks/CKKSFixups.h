/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSResultOperation.h"

NS_ASSUME_NONNULL_BEGIN

// Sometimes things go wrong.
// Sometimes you have to clean up after your past self.
// This contains the fixes.

typedef NS_ENUM(NSUInteger, CKKSFixup) {
    CKKSFixupNever,
    CKKSFixupRefetchCurrentItemPointers,
    CKKSFixupFetchTLKShares,
    CKKSFixupLocalReload,
    CKKSFixupResaveDeviceStateEntries,
    CKKSFixupDeleteAllCKKSTombstones,
};
#define CKKSCurrentFixupNumber (CKKSFixupDeleteAllCKKSTombstones)

@interface CKKSFixups : NSObject
+ (CKKSState* _Nullable)fixupOperation:(CKKSFixup)lastfixup;
@end

// Fixup declarations. You probably don't need to look at these
@interface CKKSFixupRefetchAllCurrentItemPointers : CKKSGroupOperation<OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;
- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                             ckoperationGroup:(CKOperationGroup*)ckoperationGroup;
@end

@interface CKKSFixupFetchAllTLKShares : CKKSGroupOperation<OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;
- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                             ckoperationGroup:(CKOperationGroup*)ckoperationGroup;
@end

@interface CKKSFixupLocalReloadOperation : CKKSGroupOperation<OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;
- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                                  fixupNumber:(CKKSFixup)fixupNumber
                             ckoperationGroup:(CKOperationGroup*)ckoperationGroup
                                     entering:(CKKSState*)state;
@end

@interface CKKSFixupResaveDeviceStateEntriesOperation: CKKSGroupOperation<OctagonStateTransitionOperationProtocol>
@property CKKSOperationDependencies* deps;
- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)operationDependencies;
@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
