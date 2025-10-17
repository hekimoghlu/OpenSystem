/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#import <WebKit/WKBase.h>

NS_ASSUME_NONNULL_BEGIN

@class WKWebView;
@class _WKAutomationSession;

typedef NS_ENUM(NSInteger, _WKAutomationSessionJavaScriptDialogType) {
    _WKAutomationSessionJavaScriptDialogTypeNone = 1,
    _WKAutomationSessionJavaScriptDialogTypeAlert,
    _WKAutomationSessionJavaScriptDialogTypeConfirm,
    _WKAutomationSessionJavaScriptDialogTypePrompt,
} WK_API_AVAILABLE(macos(10.14), ios(12.0));

typedef NS_ENUM(NSUInteger, _WKAutomationSessionBrowsingContextOptions) {
    _WKAutomationSessionBrowsingContextOptionsPreferNewTab = 1 << 0,
} WK_API_AVAILABLE(macos(10.14), ios(12.0));

typedef NS_ENUM(NSInteger, _WKAutomationSessionBrowsingContextPresentation) {
    _WKAutomationSessionBrowsingContextPresentationTab,
    _WKAutomationSessionBrowsingContextPresentationWindow,
} WK_API_AVAILABLE(macos(10.15), ios(13.0));

typedef NS_ENUM(NSInteger, _WKAutomationSessionWebExtensionResourceOptions) {
    _WKAutomationSessionWebExtensionResourceOptionsPath,
    _WKAutomationSessionWebExtensionResourceOptionsArchivePath,
    _WKAutomationSessionWebExtensionResourceOptionsBase64,
} WK_API_AVAILABLE(macos(WK_MAC_TBA));

@protocol _WKAutomationSessionDelegate <NSObject>
@optional

- (void)_automationSessionDidDisconnectFromRemote:(_WKAutomationSession *)automationSession;

- (void)_automationSession:(_WKAutomationSession *)automationSession requestNewWebViewWithOptions:(_WKAutomationSessionBrowsingContextOptions)options completionHandler:(void(^)(WKWebView * _Nullable))completionHandler WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession requestHideWindowOfWebView:(WKWebView *)webView completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession requestRestoreWindowOfWebView:(WKWebView *)webView completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession requestMaximizeWindowOfWebView:(WKWebView *)webView completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession requestSwitchToWebView:(WKWebView *)webView completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (BOOL)_automationSession:(_WKAutomationSession *)automationSession isShowingJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession dismissCurrentJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession acceptCurrentJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (nullable NSString *)_automationSession:(_WKAutomationSession *)automationSession messageOfCurrentJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (void)_automationSession:(_WKAutomationSession *)automationSession setUserInput:(NSString *)value forCurrentJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (_WKAutomationSessionJavaScriptDialogType)_automationSession:(_WKAutomationSession *)automationSession typeOfCurrentJavaScriptDialogForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (_WKAutomationSessionBrowsingContextPresentation)_automationSession:(_WKAutomationSession *)automationSession currentPresentationForWebView:(WKWebView *)webView WK_API_AVAILABLE(macos(10.15), ios(13.0));

- (void)_automationSession:(_WKAutomationSession *)automationSession loadWebExtensionWithOptions:(_WKAutomationSessionWebExtensionResourceOptions)options resource:(NSString *)resource completionHandler:(void(^)(NSString *))completionHandler WK_API_AVAILABLE(macos(WK_MAC_TBA));
- (void)_automationSession:(_WKAutomationSession *)automationSession unloadWebExtensionWithIdentifier:(NSString *)identifier completionHandler:(void(^)(BOOL))completionHandler WK_API_AVAILABLE(macos(WK_MAC_TBA));

@end

NS_ASSUME_NONNULL_END
