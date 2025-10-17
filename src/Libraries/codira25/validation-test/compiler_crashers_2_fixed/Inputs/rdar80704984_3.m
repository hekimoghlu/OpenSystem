/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#include "rdar80704984_3.h"

#pragma clang assume_nonnull begin

@implementation PFXObject

- (BOOL)enqueueFailingRequestWithData:(nullable NSMutableData *)data
                                error:(NSError *_Nullable *)error
                    completionHandler:
                        (nullable CompletionHandler)completionHandler {
  *error = [[NSError alloc] initWithDomain:@"d" code:1 userInfo:nil];
  return NO;
}
- (BOOL)enqueuePassingRequestWithData:(nullable NSMutableData *)data
                                error:(NSError *_Nullable *)error
                    completionHandler:
                        (nullable CompletionHandler)completionHandler {
  dispatch_async(dispatch_get_main_queue(), ^{
    completionHandler(0, 2);
  });
  return YES;
}

- (BOOL)enqueueFailingRequestWithData:(nullable NSMutableData *)data
                    completionTimeout:(NSTimeInterval)completionTimeout
                                error:(NSError *_Nullable *)error
                    completionHandler:
                        (nullable CompletionHandler)completionHandler {
  *error = [[NSError alloc] initWithDomain:@"d" code:2 userInfo:nil];
  return NO;
}
- (BOOL)enqueuePassingRequestWithData:(nullable NSMutableData *)data
                    completionTimeout:(NSTimeInterval)completionTimeout
                                error:(NSError *_Nullable *)error
                    completionHandler:
                        (nullable CompletionHandler)completionHandler {
  dispatch_async(dispatch_get_main_queue(), ^{
    completionHandler(0, 3);
  });
  return YES;
}

@end

#pragma clang assume_nonnull end
