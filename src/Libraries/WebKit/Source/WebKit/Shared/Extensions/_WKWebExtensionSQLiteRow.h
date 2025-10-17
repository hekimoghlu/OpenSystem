/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#pragma once

NS_ASSUME_NONNULL_BEGIN

@class _WKWebExtensionSQLiteRowEnumerator;
@class _WKWebExtensionSQLiteStatement;

typedef int _WKSQLiteErrorCode;

@interface _WKWebExtensionSQLiteRow : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithStatement:(_WKWebExtensionSQLiteStatement *)statement NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithCurrentRowOfEnumerator:(_WKWebExtensionSQLiteRowEnumerator *)rowEnumerator;

- (nullable NSString *)stringAtIndex:(NSUInteger)index;
- (int)intAtIndex:(NSUInteger)index;
- (int64_t)int64AtIndex:(NSUInteger)index;
- (double)doubleAtIndex:(NSUInteger)index;
- (BOOL)boolAtIndex:(NSUInteger)index;
- (nullable NSData *)dataAtIndex:(NSUInteger)index;
- (nullable NSObject *)objectAtIndex:(NSUInteger)index;

- (nullable NSData *)uncopiedDataAtIndex:(NSUInteger)index;

struct RawData {
    const bool isNull;
    const void* bytes;
    const int length;
};

- (struct RawData)uncopiedRawDataAtIndex:(NSUInteger)index;

@end

@interface _WKWebExtensionSQLiteRowEnumerator : NSEnumerator

- (instancetype)initWithResultsOfStatement:(_WKWebExtensionSQLiteStatement *)statement;

@property (readonly, nonatomic) _WKWebExtensionSQLiteStatement *statement;
@property (readonly, nonatomic) _WKSQLiteErrorCode lastResultCode;

@end

NS_ASSUME_NONNULL_END
