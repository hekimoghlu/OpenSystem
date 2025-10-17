/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include "rdar81590807_2.h"

#pragma clang assume_nonnull begin

@implementation PFXObject
- (void)findAnswerSyncSuccessAsynchronously:
    (void (^)(NSString *_Nullable, NSError *_Nullable))handler
    __attribute__((language_name("findAnswerSyncSuccess(completionHandler:)"))) {
  handler(@"syncSuccess", NULL);
}
- (void)findAnswerAsyncSuccessAsynchronously:
    (void (^)(NSString *_Nullable, NSError *_Nullable))handler
    __attribute__((language_name("findAnswerAsyncSuccess(completionHandler:)"))) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    handler(@"asyncSuccess", NULL);
  });
}
- (void)findAnswerSyncFailAsynchronously:
    (void (^)(NSString *_Nullable, NSError *_Nullable))handler
    __attribute__((language_name("findAnswerSyncFail(completionHandler:)"))) {
  handler(NULL, [NSError errorWithDomain:@"syncFail" code:1 userInfo:nil]);
}
- (void)findAnswerAsyncFailAsynchronously:
    (void (^)(NSString *_Nullable, NSError *_Nullable))handler
    __attribute__((language_name("findAnswerAsyncFail(completionHandler:)"))) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    handler(NULL, [NSError errorWithDomain:@"asyncFail" code:2 userInfo:nil]);
  });
}
- (void)findAnswerIncorrectAsynchronously:
    (void (^)(NSString *_Nullable, NSError *_Nullable))handler
    __attribute__((language_name("findAnswerIncorrect(completionHandler:)"))) {
  handler(NULL, NULL);
}
@end

#pragma clang assume_nonnull end
