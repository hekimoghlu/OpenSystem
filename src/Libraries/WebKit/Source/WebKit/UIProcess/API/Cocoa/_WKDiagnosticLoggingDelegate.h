/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

#define HAS_DIAGNOSTIC_LOGGING_DOMAIN

typedef NS_ENUM(NSInteger, _WKDiagnosticLoggingResultType) {
    _WKDiagnosticLoggingResultPass,
    _WKDiagnosticLoggingResultFail,
    _WKDiagnosticLoggingResultNoop,
} WK_API_AVAILABLE(macos(10.11), ios(9.0));

typedef NS_ENUM(NSInteger, _WKDiagnosticLoggingDomain) {
    _WKDiagnosticLoggingDomainMedia,
} WK_API_AVAILABLE(macos(12.0), ios(15.0));

@class WKWebView;

@protocol _WKDiagnosticLoggingDelegate <NSObject>
@optional

- (void)_webView:(WKWebView *)webView logDiagnosticMessage:(NSString *)message description:(NSString *)description;
- (void)_webView:(WKWebView *)webView logDiagnosticMessageWithResult:(NSString *)message description:(NSString *)description result:(_WKDiagnosticLoggingResultType)result;
- (void)_webView:(WKWebView *)webView logDiagnosticMessageWithValue:(NSString *)message description:(NSString *)description value:(NSString *) value;
- (void)_webView:(WKWebView *)webView logDiagnosticMessageWithEnhancedPrivacy:(NSString *)message description:(NSString *)description WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (void)_webView:(WKWebView *)webView logDiagnosticMessage:(NSString *)message description:(NSString *)description valueDictionary:(NSDictionary *)valueDictionary WK_API_AVAILABLE(macos(10.15), ios(13.0));
- (void)_webView:(WKWebView *)webView logDiagnosticMessageWithDomain:(NSString *)message domain:(_WKDiagnosticLoggingDomain)domain WK_API_AVAILABLE(macos(12.0), ios(15.0));

@end
