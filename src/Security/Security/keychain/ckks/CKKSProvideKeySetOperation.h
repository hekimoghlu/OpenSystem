/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#import <CloudKit/CloudKit.h>

#import "keychain/ckks/CKKSCurrentKeyPointer.h"
#import "keychain/ckks/CKKSGroupOperation.h"

NS_ASSUME_NONNULL_BEGIN

@protocol CKKSKeySetContainerProtocol <NSObject>
@property (readonly, nullable) NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>* keysets;

// This contains the list of views that we intended to fetch
@property (readonly) NSSet<CKRecordZoneID*>* intendedZoneIDs;
@end

@protocol CKKSKeySetProviderOperationProtocol <NSObject, CKKSKeySetContainerProtocol>
- (void)provideKeySets:(NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>*)keysets;
@end

// This is an odd operation:
//   If you call init: and then add the operation to a queue, it will not start until provideKeySet runs.
//     But! -timeout: will work, and the operation will finish
@interface CKKSProvideKeySetOperation : CKKSGroupOperation <CKKSKeySetProviderOperationProtocol>
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithIntendedZoneIDs:(NSSet<CKRecordZoneID*>*)intendedZones;

- (void)provideKeySets:(NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>*)keysets;
@end

NS_ASSUME_NONNULL_END

#endif
