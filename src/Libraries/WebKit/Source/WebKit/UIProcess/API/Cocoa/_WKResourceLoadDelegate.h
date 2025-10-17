/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

@class WKWebView;
@class _WKResourceLoadInfo;

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(11.0), ios(14.0))
@protocol _WKResourceLoadDelegate <NSObject>

@optional
- (void)webView:(WKWebView *)webView resourceLoad:(_WKResourceLoadInfo *)resourceLoad didSendRequest:(NSURLRequest *)request;
- (void)webView:(WKWebView *)webView resourceLoad:(_WKResourceLoadInfo *)resourceLoad didPerformHTTPRedirection:(NSURLResponse *)response newRequest:(NSURLRequest *)request;
- (void)webView:(WKWebView *)webView resourceLoad:(_WKResourceLoadInfo *)resourceLoad didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge;
- (void)webView:(WKWebView *)webView resourceLoad:(_WKResourceLoadInfo *)resourceLoad didReceiveResponse:(NSURLResponse *)response;
- (void)webView:(WKWebView *)webView resourceLoad:(_WKResourceLoadInfo *)resourceLoad didCompleteWithError:(nullable NSError *)error response:(nullable NSURLResponse *)response;

@end

NS_ASSUME_NONNULL_END
