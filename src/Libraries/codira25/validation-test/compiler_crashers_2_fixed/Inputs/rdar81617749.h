/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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

typedef void (^CompletionHandler)(void);

@interface PFXObject : NSObject
- (void)performSingleFlaggy1WithCompletionHandler:
    (void (^)(BOOL, CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performSingleFlaggy2WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));

- (void)performSingleErrory1WithCompletionHandler:
    (void (^)(NSError *_Nullable,
              CompletionHandler _Nullable))completionHandler;
- (void)performSingleErrory2WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable,
              NSError *_Nullable))completionHandler;

- (void)performSingleBothy12WithCompletionHandler:
    (void (^)(NSError *_Nullable, BOOL,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));
- (void)performSingleBothy13WithCompletionHandler:
    (void (^)(NSError *_Nullable, CompletionHandler _Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));
- (void)performSingleBothy21WithCompletionHandler:
    (void (^)(BOOL, NSError *_Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performSingleBothy23WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, NSError *_Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));
- (void)performSingleBothy31WithCompletionHandler:
    (void (^)(BOOL, CompletionHandler _Nullable,
              NSError *_Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performSingleBothy32WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, BOOL,
              NSError *_Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));

- (void)performDoubleFlaggy1WithCompletionHandler:
    (void (^)(BOOL, CompletionHandler _Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performDoubleFlaggy2WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, BOOL,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));
- (void)performDoubleFlaggy3WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, CompletionHandler _Nullable,
              BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));

- (void)performDoubleErrory1WithCompletionHandler:
    (void (^)(NSError *_Nullable, CompletionHandler _Nullable,
              CompletionHandler _Nullable))completionHandler;
- (void)performDoubleErrory2WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, NSError *_Nullable,
              CompletionHandler _Nullable))completionHandler;
- (void)performDoubleErrory3WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, CompletionHandler _Nullable,
              NSError *_Nullable))completionHandler;

- (void)performDoubleBothy12WithCompletionHandler:
    (void (^)(NSError *_Nullable, BOOL, CompletionHandler _Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));
- (void)performDoubleBothy13WithCompletionHandler:
    (void (^)(NSError *_Nullable, CompletionHandler _Nullable, BOOL,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));
- (void)performDoubleBothy14WithCompletionHandler:
    (void (^)(NSError *_Nullable, CompletionHandler _Nullable,
              CompletionHandler _Nullable, BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 4)));

- (void)performDoubleBothy21WithCompletionHandler:
    (void (^)(BOOL, NSError *_Nullable, CompletionHandler _Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performDoubleBothy23WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, NSError *_Nullable, BOOL,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));
- (void)performDoubleBothy24WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, NSError *_Nullable,
              CompletionHandler _Nullable, BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 4)));

- (void)performDoubleBothy31WithCompletionHandler:
    (void (^)(BOOL, CompletionHandler _Nullable, NSError *_Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performDoubleBothy32WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, BOOL, NSError *_Nullable,
              CompletionHandler _Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));
- (void)performDoubleBothy34WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, CompletionHandler _Nullable,
              NSError *_Nullable, BOOL))completionHandler
    __attribute__((language_async_error(zero_argument, 4)));

- (void)performDoubleBothy41WithCompletionHandler:
    (void (^)(BOOL, CompletionHandler _Nullable, CompletionHandler _Nullable,
              NSError *_Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 1)));
- (void)performDoubleBothy42WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, BOOL, CompletionHandler _Nullable,
              NSError *_Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 2)));
- (void)performDoubleBothy43WithCompletionHandler:
    (void (^)(CompletionHandler _Nullable, CompletionHandler _Nullable, BOOL,
              NSError *_Nullable))completionHandler
    __attribute__((language_async_error(zero_argument, 3)));
@end

#pragma clang assume_nonnull end
