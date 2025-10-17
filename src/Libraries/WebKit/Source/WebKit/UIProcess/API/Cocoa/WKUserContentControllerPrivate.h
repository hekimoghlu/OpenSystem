/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#import <WebKit/WKUserContentController.h>

@class WKContentWorld;
@class WKUserScript;
@class _WKUserContentFilter;
@class _WKUserContentWorld;
@class _WKUserStyleSheet;

@interface WKUserContentController (WKPrivate)

- (void)_removeUserScript:(WKUserScript *)userScript WK_API_AVAILABLE(macos(10.12), ios(10.0));
- (void)_removeAllUserScriptsAssociatedWithContentWorld:(WKContentWorld *)contentWorld WK_API_AVAILABLE(macos(11.0), ios(14.0));

- (void)_addUserScriptImmediately:(WKUserScript *)userScript WK_API_AVAILABLE(macos(10.14), ios(12.0));

// FIXME: Remove this once rdar://100785999 is unblocked.
- (void)_addUserContentFilter:(_WKUserContentFilter *)userContentFilter WK_API_DEPRECATED_WITH_REPLACEMENT("addContentRuleList", macos(10.11, 13.3), ios(9.0, 16.4));

- (void)_removeUserContentFilter:(NSString *)userContentFilterName WK_API_AVAILABLE(macos(10.11), ios(9.0));
- (void)_removeAllUserContentFilters WK_API_AVAILABLE(macos(10.11), ios(9.0));
- (void)_addContentRuleList:(WKContentRuleList *)contentRuleList extensionBaseURL:(NSURL *)extensionBaseURL WK_API_AVAILABLE(macos(13.0), ios(16.0));

@property (nonatomic, readonly, copy) NSArray<_WKUserStyleSheet *> *_userStyleSheets WK_API_AVAILABLE(macos(10.12), ios(10.0));
- (void)_addUserStyleSheet:(_WKUserStyleSheet *)userStyleSheet WK_API_AVAILABLE(macos(10.12), ios(10.0));
- (void)_removeUserStyleSheet:(_WKUserStyleSheet *)userStyleSheet WK_API_AVAILABLE(macos(10.12), ios(10.0));
- (void)_removeAllUserStyleSheets WK_API_AVAILABLE(macos(10.12), ios(10.0));
- (void)_removeAllUserStyleSheetsAssociatedWithContentWorld:(WKContentWorld *)contentWorld WK_API_AVAILABLE(macos(11.0), ios(14.0));

- (void)_addScriptMessageHandler:(id <WKScriptMessageHandler>)scriptMessageHandler name:(NSString *)name userContentWorld:(_WKUserContentWorld *)userContentWorld WK_API_DEPRECATED_WITH_REPLACEMENT("_addScriptMessageHandler:name:contentWorld:", macos(10.11, 11.0), ios(9.0, 14.0));
- (void)_addScriptMessageHandler:(id <WKScriptMessageHandler>)scriptMessageHandler name:(NSString *)name contentWorld:(WKContentWorld *)contentWorld;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
- (void)_removeScriptMessageHandlerForName:(NSString *)name userContentWorld:(_WKUserContentWorld *)userContentWorld;
- (void)_removeAllScriptMessageHandlersAssociatedWithUserContentWorld:(_WKUserContentWorld *)userContentWorld;
#pragma clang diagnostic pop

@end
