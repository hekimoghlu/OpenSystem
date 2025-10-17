/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#import "keychain/ckks/CKKSKeychainView.h"
NS_ASSUME_NONNULL_BEGIN


@interface CKKSUpdateCurrentItemPointerOperation : CKKSGroupOperation

@property NSString* currentPointerIdentifier;
@property (readonly) CKKSKeychainViewState* viewState;

- (instancetype)init NS_UNAVAILABLE;

// Unlike many other CKKS operations, this one takes a specific zone to operate in.
- (instancetype)initWithCKKSOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                                        viewState:(CKKSKeychainViewState*)viewState
                                          newItem:(NSData*)newItemPersistentRef
                                             hash:(NSData*)newItemSHA1
                                      accessGroup:(NSString*)accessGroup
                                       identifier:(NSString*)identifier
                                        replacing:(NSData* _Nullable)oldCurrentItemPersistentRef
                                             hash:(NSData* _Nullable)oldItemSHA1
                                 ckoperationGroup:(CKOperationGroup* _Nullable)ckoperationGroup;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
