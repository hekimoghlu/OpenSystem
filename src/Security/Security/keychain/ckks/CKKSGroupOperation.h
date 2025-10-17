/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
#include <dispatch/dispatch.h>
#import "keychain/ckks/CKKSResultOperation.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSGroupOperation : CKKSResultOperation
{
    BOOL executing;
    BOOL finished;
}

+ (instancetype)operationWithBlock:(void (^)(void))block;
+ (instancetype)named:(NSString*)name withBlock:(void (^)(void))block;
+ (instancetype)named:(NSString*)name withBlockTakingSelf:(void(^)(CKKSResultOperation* strongOp))block;

@property NSOperationQueue* operationQueue;

- (instancetype)init;

// For subclasses: override this to execute at Group operation start time
- (void)groupStart;

- (void)runBeforeGroupFinished:(NSOperation*)suboperation;
- (void)dependOnBeforeGroupFinished:(NSOperation*)suboperation;
@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
