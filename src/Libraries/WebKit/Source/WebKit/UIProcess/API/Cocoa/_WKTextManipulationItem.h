/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

@class _WKTextManipulationToken;

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(10.15.4), ios(13.4))
@interface _WKTextManipulationItem : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithIdentifier:(nullable NSString *)identifier tokens:(NSArray<_WKTextManipulationToken *> *)tokens;
- (instancetype)initWithIdentifier:(nullable NSString *)identifier tokens:(NSArray<_WKTextManipulationToken *> *)tokens isSubframe:(BOOL)isSubframe isCrossSiteSubframe:(BOOL)isCrossSiteSubframe WK_API_AVAILABLE(macos(14.0), ios(17.0));

@property (nonatomic, readonly, nullable, copy) NSString *identifier;
@property (nonatomic, readonly, copy) NSArray<_WKTextManipulationToken *> *tokens;
@property (nonatomic, readonly) BOOL isSubframe WK_API_AVAILABLE(macos(14.0), ios(17.0));
@property (nonatomic, readonly) BOOL isCrossSiteSubframe WK_API_AVAILABLE(macos(14.0), ios(17.0));

- (BOOL)isEqualToTextManipulationItem:(nullable _WKTextManipulationItem *)otherItem includingContentEquality:(BOOL)includingContentEquality;
@property (nonatomic, readonly, copy) NSString *debugDescription;

@end

WK_EXTERN NSString * const _WKTextManipulationItemErrorDomain WK_API_AVAILABLE(macos(11.0), ios(14.0));

typedef NS_ENUM(NSInteger, _WKTextManipulationItemErrorCode) {
    _WKTextManipulationItemErrorNotAvailable,
    _WKTextManipulationItemErrorContentChanged,
    _WKTextManipulationItemErrorInvalidItem,
    _WKTextManipulationItemErrorInvalidToken,
    _WKTextManipulationItemErrorExclusionViolation,
} WK_API_AVAILABLE(macos(11.0), ios(14.0));

WK_EXTERN NSErrorUserInfoKey const _WKTextManipulationItemErrorItemKey WK_API_AVAILABLE(macos(11.0), ios(14.0));

NS_ASSUME_NONNULL_END
