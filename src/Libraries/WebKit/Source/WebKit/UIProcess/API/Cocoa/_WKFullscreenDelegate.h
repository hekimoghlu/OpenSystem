/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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

NS_ASSUME_NONNULL_BEGIN

@class NSError;
@class UIViewController;

WK_API_AVAILABLE(macos(10.13), ios(11.3))
@protocol _WKFullscreenDelegate <NSObject>

@optional

#if TARGET_OS_IPHONE
- (void)_webViewWillEnterElementFullscreen:(WKWebView *)webView;
- (void)_webViewDidEnterElementFullscreen:(WKWebView *)webView;
- (void)_webViewWillExitElementFullscreen:(WKWebView *)webView;
- (void)_webViewDidExitElementFullscreen:(WKWebView *)webView;

- (void)_webView:(WKWebView *)webView didFullscreenImageWithQuickLook:(CGSize)imageDimensions;
- (void)_webView:(WKWebView *)webView requestPresentingViewControllerWithCompletionHandler:(void (^)(UIViewController * _Nullable, NSError * _Nullable))completionHandler;
#else
- (void)_webViewWillEnterFullscreen:(NSView *)webView;
- (void)_webViewDidEnterFullscreen:(NSView *)webView;
- (void)_webViewWillExitFullscreen:(NSView *)webView;
- (void)_webViewDidExitFullscreen:(NSView *)webView;
#endif

@end

NS_ASSUME_NONNULL_END
