/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

NS_ASSUME_NONNULL_BEGIN

WK_EXTERN NSString * const _WKTextManipulationTokenUserInfoDocumentURLKey WK_API_AVAILABLE(macos(11.0), ios(14.0));
WK_EXTERN NSString * const _WKTextManipulationTokenUserInfoTagNameKey WK_API_AVAILABLE(macos(11.0), ios(14.0));
WK_EXTERN NSString * const _WKTextManipulationTokenUserInfoRoleAttributeKey WK_API_AVAILABLE(macos(11.0), ios(14.0));
WK_EXTERN NSString * const _WKTextManipulationTokenUserInfoVisibilityKey WK_API_AVAILABLE(macos(12.0), ios(15.0));

WK_CLASS_AVAILABLE(macos(10.15.4), ios(13.4))
@interface _WKTextManipulationToken : NSObject

@property (nonatomic, nullable, copy) NSString *identifier;
@property (nonatomic, nullable, copy) NSString *content;
@property (nonatomic, getter=isExcluded) BOOL excluded;

- (BOOL)isEqualToTextManipulationToken:(nullable _WKTextManipulationToken *)otherToken includingContentEquality:(BOOL)includingContentEquality;
@property (nonatomic, copy, readonly) NSString *debugDescription;

@property (nonatomic, nullable, copy) NSDictionary<NSString *, id> *userInfo;

@end

NS_ASSUME_NONNULL_END
