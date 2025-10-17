/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#include <Foundation/Foundation.h>

#pragma clang assume_nonnull begin

@interface PFXObject : NSObject {
}
- (BOOL)failReturnWithError:(NSError *_Nullable *)error
          completionHandler:
              (void (^_Nonnull)(NSError *_Nullable error))completionHandler;
- (BOOL)failInvokeSyncWithError:(NSError *_Nullable *)error
              completionHandler:
                  (void (^_Nonnull)(NSError *_Nullable error))completionHandler;
- (BOOL)failInvokeAsyncWithError:(NSError *_Nullable *)error
               completionHandler:(void (^_Nonnull)(NSError *_Nullable error))
                                     completionHandler;
- (BOOL)succeedSyncWithError:(NSError *_Nullable *)error
           completionHandler:
               (void (^_Nonnull)(NSError *_Nullable error))completionHandler;
- (BOOL)succeedAsyncWithError:(NSError *_Nullable *)error
            completionHandler:
                (void (^_Nonnull)(NSError *_Nullable error))completionHandler;
@end

#pragma clang assume_nonnull end
