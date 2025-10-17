/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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

@class _WKUserContentWorld;
@class WKContentWorld;
@class WKWebView;

typedef NS_ENUM(NSInteger, _WKUserStyleLevel) {
    _WKUserStyleUserLevel,
    _WKUserStyleAuthorLevel
} WK_API_AVAILABLE(macos(11.0), ios(14.0));

WK_CLASS_AVAILABLE(macos(10.12), ios(10.0))
@interface _WKUserStyleSheet : NSObject <NSCopying>

@property (nonatomic, readonly, copy) NSString *source;

@property (nonatomic, readonly, copy) NSURL *baseURL;

@property (nonatomic, readonly, getter=isForMainFrameOnly) BOOL forMainFrameOnly;

- (instancetype)initWithSource:(NSString *)source forMainFrameOnly:(BOOL)forMainFrameOnly;

- (instancetype)initWithSource:(NSString *)source forWKWebView:(nullable WKWebView *)webView forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(nullable NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(nullable NSArray<NSString *> *)excludeMatchPatternStrings baseURL:(nullable NSURL *)baseURL level:(_WKUserStyleLevel)level contentWorld:(nullable WKContentWorld *)contentWorld WK_API_AVAILABLE(macos(11.0), ios(14.0));

@end

NS_ASSUME_NONNULL_END
