/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#import <WebKit/WKUserScript.h>

NS_ASSUME_NONNULL_BEGIN

@class WKContentWorld;
@class _WKUserContentWorld;

@interface WKUserScript (WKPrivate)

- (instancetype)_initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(nullable NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(nullable NSArray<NSString *> *)excludeMatchPatternStrings associatedURL:(nullable NSURL *)associatedURL contentWorld:(nullable WKContentWorld *)contentWorld WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));
- (instancetype)_initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly includeMatchPatternStrings:(nullable NSArray<NSString *> *)includeMatchPatternStrings excludeMatchPatternStrings:(nullable NSArray<NSString *> *)excludeMatchPatternStrings associatedURL:(nullable NSURL *)associatedURL contentWorld:(nullable WKContentWorld *)contentWorld deferRunningUntilNotification:(BOOL)deferRunningUntilNotification WK_API_DEPRECATED_WITH_REPLACEMENT("-_initWithSource:injectionTime:forMainFrameOnly:includeMatchPatternStrings:excludeMatchPatternStrings:associatedURL:contentWorld:", macos(11.0, WK_MAC_TBA), ios(14.0, WK_IOS_TBA), visionos(1.0, WK_XROS_TBA));

@property (nonatomic, readonly) _WKUserContentWorld *_userContentWorld WK_API_DEPRECATED_WITH_REPLACEMENT("_contentWorld", macos(10.12, 11.0), ios(10.0, 14.0));
@property (nonatomic, readonly) WKContentWorld *_contentWorld WK_API_AVAILABLE(macos(11.0), ios(14.0));

@end

NS_ASSUME_NONNULL_END
