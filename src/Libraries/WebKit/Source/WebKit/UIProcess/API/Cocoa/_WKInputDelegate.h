/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

@class UITextSuggestion;
@class WKWebView;

@protocol _WKFocusedElementInfo;
@protocol _WKFormInputSession;

#if TARGET_OS_IPHONE
typedef NS_ENUM(NSInteger, _WKFocusStartsInputSessionPolicy) {
    _WKFocusStartsInputSessionPolicyAuto,
    _WKFocusStartsInputSessionPolicyAllow,
    _WKFocusStartsInputSessionPolicyDisallow,
} WK_API_AVAILABLE(ios(12.0));
#endif

@protocol _WKInputDelegate <NSObject>

@optional

- (void)_webView:(WKWebView *)webView didStartInputSession:(id <_WKFormInputSession>)inputSession;
- (void)_webView:(WKWebView *)webView willSubmitFormValues:(NSDictionary *)values userObject:(NSObject <NSSecureCoding> *)userObject submissionHandler:(void (^)(void))submissionHandler;

#if TARGET_OS_IPHONE
- (BOOL)_webView:(WKWebView *)webView focusShouldStartInputSession:(id <_WKFocusedElementInfo>)info;
- (_WKFocusStartsInputSessionPolicy)_webView:(WKWebView *)webView decidePolicyForFocusedElement:(id <_WKFocusedElementInfo>)info WK_API_AVAILABLE(ios(12.0));
- (void)_webView:(WKWebView *)webView accessoryViewCustomButtonTappedInFormInputSession:(id <_WKFormInputSession>)inputSession;
- (void)_webView:(WKWebView *)webView insertTextSuggestion:(UITextSuggestion *)suggestion inInputSession:(id <_WKFormInputSession>)inputSession WK_API_AVAILABLE(ios(10.0));
- (void)_webView:(WKWebView *)webView willStartInputSession:(id <_WKFormInputSession>)inputSession WK_API_AVAILABLE(ios(12.0));
- (BOOL)_webView:(WKWebView *)webView focusRequiresStrongPasswordAssistance:(id <_WKFocusedElementInfo>)info WK_API_AVAILABLE(ios(12.0));

- (NSDictionary<id, NSString *> *)_webViewAdditionalContextForStrongPasswordAssistance:(WKWebView *)webView WK_API_AVAILABLE(ios(13.4));

- (BOOL)_webView:(WKWebView *)webView shouldRevealFocusOverlayForInputSession:(id <_WKFormInputSession>)inputSession WK_API_AVAILABLE(ios(12.0));
- (CGFloat)_webView:(WKWebView *)webView focusedElementContextViewHeightForFittingSize:(CGSize)fittingSize inputSession:(id <_WKFormInputSession>)inputSession WK_API_AVAILABLE(ios(12.0));
- (UIView *)_webView:(WKWebView *)webView focusedElementContextViewForInputSession:(id <_WKFormInputSession>)inputSession WK_API_AVAILABLE(ios(12.0));
#endif

@end
