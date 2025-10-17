/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

#include "rdar81590807.h"

#pragma clang assume_nonnull begin

@implementation PFXObject
- (void)continuePassSyncWithCompletionHandler:(void (^)(void (^_Nullable)(void),
                                                        NSError *_Nullable,
                                                        BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3))) {
  completionHandler(
      ^{
        NSLog(@"passSync");
      },
      NULL, YES);
}
- (void)continuePassAsyncWithCompletionHandler:
    (void (^)(void (^_Nullable)(void), NSError *_Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3))) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    completionHandler(
        ^{
          NSLog(@"passAsync");
        },
        NULL, YES);
  });
}
- (void)continueFailSyncWithCompletionHandler:(void (^)(void (^_Nullable)(void),
                                                        NSError *_Nullable,
                                                        BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3))) {
  completionHandler(
      NULL, [NSError errorWithDomain:@"failSync" code:1 userInfo:nil], NO);
}
- (void)continueFailAsyncWithCompletionHandler:
    (void (^)(void (^_Nullable)(void), NSError *_Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3))) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    completionHandler(
        NULL, [NSError errorWithDomain:@"failAsync" code:2 userInfo:nil], NO);
  });
}
- (void)continueIncorrectWithCompletionHandler:
    (void (^)(void (^_Nullable)(void), NSError *_Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3))) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    completionHandler(NULL, NULL, NO);
  });
}
@end

#pragma clang assume_nonnull end
