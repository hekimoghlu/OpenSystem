/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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

#include "rdar85526916.h"

#pragma clang assume_nonnull begin

@implementation PFXObject
- (void)performGetStringIdentityWithCompletionHandler:
    (void (^)(NSString * _Nonnull(^ _Nonnull)(NSString * _Nonnull)))completionHandler {
  completionHandler(^(NSString * _Nonnull input) {
    return input;
  });
}
- (void)performGetStringAppendWithCompletionHandler:
    (void (^)(NSString * _Nonnull(^ _Nonnull)(NSString * _Nonnull, NSString * _Nonnull)))completionHandler {
  completionHandler(^(NSString * _Nonnull one, NSString * _Nonnull two) {
    return [one stringByAppendingString: two];
  });
}
- (void)performGetIntegerIdentityWithCompletionHandler:
    (void (^)(NSInteger(^ _Nonnull)(NSInteger)))completionHandler {
  completionHandler(^(NSInteger input) {
    return input;
  });
}
- (void)performGetIntegerSubtractWithCompletionHandler:
    (void (^)(NSInteger(^ _Nonnull)(NSInteger, NSInteger)))completionHandler {
  completionHandler(^(NSInteger lhs, NSInteger rhs) {
    return lhs - rhs;
  });
}
- (void)performGetUIntegerIdentityWithCompletionHandler:
    (void (^)(NSUInteger(^ _Nonnull)(NSUInteger)))completionHandler {
  completionHandler(^(NSUInteger input) {
    return input;
  });
}
- (void)performGetUIntegerAddWithCompletionHandler:
    (void (^)(NSUInteger(^ _Nonnull)(NSUInteger, NSUInteger)))completionHandler {
  completionHandler(^(NSUInteger lhs, NSUInteger rhs) {
    return lhs + rhs;
  });
}
@end

#pragma clang assume_nonnull end

