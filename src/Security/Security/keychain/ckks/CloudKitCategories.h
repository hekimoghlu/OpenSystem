/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#import <CloudKit/CloudKit.h>
#import <CloudKit/CloudKit_Private.h>
#import <Foundation/Foundation.h>

#import "keychain/ot/OTDefines.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKOperationGroup (CKKS)
+ (instancetype)CKKSGroupWithName:(NSString*)name;
@end

@interface NSError (CKKS)
// Returns true if this is a CloudKit error where
// 1) An atomic write failed
// 2) Every single suberror is either CKErrorServerRecordChanged or CKErrorUnknownItem
- (bool)ckksIsCKErrorRecordChangedError;

- (BOOL)isCKKSServerPluginError:(NSInteger)code;
- (BOOL)isCKServerInternalError;
- (BOOL)isCKInternalServerHTTPError;
@end
// Ensure we don't print addresses
@interface CKAccountInfo (CKKS)
- (NSString*)description;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
