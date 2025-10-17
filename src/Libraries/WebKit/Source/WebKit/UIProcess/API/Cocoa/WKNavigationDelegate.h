/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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

NS_ASSUME_NONNULL_BEGIN

@class WKBackForwardListItem;
@class WKDownload;
@class WKNavigation;
@class WKNavigationAction;
@class WKNavigationResponse;
@class WKWebView;
@class WKWebpagePreferences;

/*! @enum WKNavigationActionPolicy
 @abstract The policy to pass back to the decision handler from the
 webView:decidePolicyForNavigationAction:decisionHandler: method.
 @constant WKNavigationActionPolicyCancel   Cancel the navigation.
 @constant WKNavigationActionPolicyAllow    Allow the navigation to continue.
 @constant WKNavigationActionPolicyDownload    Turn the navigation into a download.
 */
typedef NS_ENUM(NSInteger, WKNavigationActionPolicy) {
    WKNavigationActionPolicyCancel,
    WKNavigationActionPolicyAllow,
    WKNavigationActionPolicyDownload WK_API_AVAILABLE(macos(11.3), ios(14.5)),
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

/*! @enum WKNavigationResponsePolicy
 @abstract The policy to pass back to the decision handler from the webView:decidePolicyForNavigationResponse:decisionHandler: method.
 @constant WKNavigationResponsePolicyCancel   Cancel the navigation.
 @constant WKNavigationResponsePolicyAllow    Allow the navigation to continue.
 @constant WKNavigationResponsePolicyDownload    Turn the navigation into a download.
 */
typedef NS_ENUM(NSInteger, WKNavigationResponsePolicy) {
    WKNavigationResponsePolicyCancel,
    WKNavigationResponsePolicyAllow,
    WKNavigationResponsePolicyDownload WK_API_AVAILABLE(macos(11.3), ios(14.5)),
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

/*! A class conforming to the WKNavigationDelegate protocol can provide
 methods for tracking progress for main frame navigations and for deciding
 policy for main frame and subframe navigations.
 */
WK_SWIFT_UI_ACTOR
@protocol WKNavigationDelegate <NSObject>

@optional

/*! @abstract Decides whether to allow or cancel a navigation.
 @param webView The web view invoking the delegate method.
 @param navigationAction Descriptive information about the action
 triggering the navigation request.
 @param decisionHandler The decision handler to call to allow or cancel the
 navigation. The argument is one of the constants of the enumerated type WKNavigationActionPolicy.
 @discussion If you do not implement this method, the web view will load the request or, if appropriate, forward it to another application.
 */
- (void)webView:(WKWebView *)webView decidePolicyForNavigationAction:(WKNavigationAction *)navigationAction decisionHandler:(WK_SWIFT_UI_ACTOR void (^)(WKNavigationActionPolicy))decisionHandler WK_SWIFT_ASYNC(3);

/*! @abstract Decides whether to allow or cancel a navigation.
 @param webView The web view invoking the delegate method.
 @param navigationAction Descriptive information about the action
 triggering the navigation request.
 @param preferences The default set of webpage preferences. This may be
 changed by setting defaultWebpagePreferences on WKWebViewConfiguration.
 @param decisionHandler The policy decision handler to call to allow or cancel
 the navigation. The arguments are one of the constants of the enumerated type
 WKNavigationActionPolicy, as well as an instance of WKWebpagePreferences.
 @discussion If you implement this method,
 -webView:decidePolicyForNavigationAction:decisionHandler: will not be called.
 */
- (void)webView:(WKWebView *)webView decidePolicyForNavigationAction:(WKNavigationAction *)navigationAction preferences:(WKWebpagePreferences *)preferences decisionHandler:(WK_SWIFT_UI_ACTOR void (^)(WKNavigationActionPolicy, WKWebpagePreferences *))decisionHandler WK_SWIFT_ASYNC(4) WK_API_AVAILABLE(macos(10.15), ios(13.0));

/*! @abstract Decides whether to allow or cancel a navigation after its
 response is known.
 @param webView The web view invoking the delegate method.
 @param navigationResponse Descriptive information about the navigation
 response.
 @param decisionHandler The decision handler to call to allow or cancel the
 navigation. The argument is one of the constants of the enumerated type WKNavigationResponsePolicy.
 @discussion If you do not implement this method, the web view will allow the response, if the web view can show it.
 */
- (void)webView:(WKWebView *)webView decidePolicyForNavigationResponse:(WKNavigationResponse *)navigationResponse decisionHandler:(WK_SWIFT_UI_ACTOR void (^)(WKNavigationResponsePolicy))decisionHandler WK_SWIFT_ASYNC(3);

/*! @abstract Invoked when a main frame navigation starts.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 */
- (void)webView:(WKWebView *)webView didStartProvisionalNavigation:(null_unspecified WKNavigation *)navigation;

/*! @abstract Invoked when a server redirect is received for the main
 frame.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 */
- (void)webView:(WKWebView *)webView didReceiveServerRedirectForProvisionalNavigation:(null_unspecified WKNavigation *)navigation;

/*! @abstract Invoked when an error occurs while starting to load data for
 the main frame.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 @param error The error that occurred.
 */
- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(null_unspecified WKNavigation *)navigation withError:(NSError *)error;

/*! @abstract Invoked when content starts arriving for the main frame.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 */
- (void)webView:(WKWebView *)webView didCommitNavigation:(null_unspecified WKNavigation *)navigation;

/*! @abstract Invoked when a main frame navigation completes.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 */
- (void)webView:(WKWebView *)webView didFinishNavigation:(null_unspecified WKNavigation *)navigation;

/*! @abstract Invoked when an error occurs during a committed main frame
 navigation.
 @param webView The web view invoking the delegate method.
 @param navigation The navigation.
 @param error The error that occurred.
 */
- (void)webView:(WKWebView *)webView didFailNavigation:(null_unspecified WKNavigation *)navigation withError:(NSError *)error;

/*! @abstract Invoked when the web view needs to respond to an authentication challenge.
 @param webView The web view that received the authentication challenge.
 @param challenge The authentication challenge.
 @param completionHandler The completion handler you must invoke to respond to the challenge. The
 disposition argument is one of the constants of the enumerated type
 NSURLSessionAuthChallengeDisposition. When disposition is NSURLSessionAuthChallengeUseCredential,
 the credential argument is the credential to use, or nil to indicate continuing without a
 credential.
 @discussion If you do not implement this method, the web view will respond to the authentication challenge with the NSURLSessionAuthChallengeRejectProtectionSpace disposition.
 */
- (void)webView:(WKWebView *)webView didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(WK_SWIFT_UI_ACTOR void (^)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential * _Nullable credential))completionHandler WK_SWIFT_ASYNC_NAME(webView(_:respondTo:));

/*! @abstract Invoked when the web view's web content process is terminated.
 @param webView The web view whose underlying web content process was terminated.
 */
- (void)webViewWebContentProcessDidTerminate:(WKWebView *)webView WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @abstract Invoked when the web view is establishing a network connection using a deprecated version of TLS.
 @param webView The web view initiating the connection.
 @param challenge The authentication challenge.
 @param decisionHandler The decision handler you must invoke to respond to indicate whether or not to continue with the connection establishment.
 */
- (void)webView:(WKWebView *)webView authenticationChallenge:(NSURLAuthenticationChallenge *)challenge shouldAllowDeprecatedTLS:(WK_SWIFT_UI_ACTOR void (^)(BOOL))decisionHandler WK_SWIFT_ASYNC_NAME(webView(_:shouldAllowDeprecatedTLSFor:)) WK_SWIFT_ASYNC(3) WK_API_AVAILABLE(macos(11.0), ios(14.0));

/*
 @abstract Called after using WKNavigationActionPolicyDownload.
 @param webView The web view that created the download.
 @param navigationAction The action that is being turned into a download.
 @param download The download.
 @discussion The download needs its delegate to be set to receive updates about its progress.
*/
- (void)webView:(WKWebView *)webView navigationAction:(WKNavigationAction *)navigationAction didBecomeDownload:(WKDownload *)download WK_API_AVAILABLE(macos(11.3), ios(14.5));

/*
 @abstract Called after using WKNavigationResponsePolicyDownload.
 @param webView The web view that created the download.
 @param navigationResponse The response that is being turned into a download.
 @param download The download.
 @discussion The download needs its delegate to be set to receive updates about its progress.
*/
- (void)webView:(WKWebView *)webView navigationResponse:(WKNavigationResponse *)navigationResponse didBecomeDownload:(WKDownload *)download WK_API_AVAILABLE(macos(11.3), ios(14.5));

/*
 @abstract Called when the webpage initiates a back/forward navigation via JavaScript
 @param webView The web view invoking the delegate method.
 @param backForwardListItem The back/forward list item that will be navigated to
 @param willUseInstantBack Whether or not the navigation will resume a previously suspended webpage that is eligible for Instant Back
 @param completionHandler The completion handler you must invoke to allow or disallow the navigation
 @discussion Back/forward navigations - including those triggered by webpage JavaScript - will consult the WebKit client via this delegate.
 If the `willUseInstantBack` argument is `YES`, then the navigation is to a webpage that is suspended in memory and might be resumed without
 the normal webpage loading process.
 Even if the `willUseInstantBack` argument is `YES`, it is possible that the suspended webpage will not be used and the normal loading
 process will take place.
 In the case where the normal webpage loading process takes place, additional navigation delegate calls will continue to happen for this
 navigation starting with `decidePolicyForNavigationAction`
*/
- (void)webView:(WKWebView *)webView shouldGoToBackForwardListItem:(WKBackForwardListItem *)backForwardListItem willUseInstantBack:(BOOL)willUseInstantBack completionHandler:(void (^)(BOOL shouldGoToItem))completionHandler WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

@end

NS_ASSUME_NONNULL_END
