/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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

@class UNNotification;
@class WKSecurityOrigin;
@class WKWebsiteDataStore;
@class WKWebView;
@class _WKNotificationData;

/*! @enum WKNavigationActionPolicy
 @abstract The policy to pass back to the decision handler from the
 webView:decidePolicyForNavigationAction:decisionHandler: method.
 @constant WKNavigationActionPolicyCancel   Cancel the navigation.
 @constant WKNavigationActionPolicyAllow    Allow the navigation to continue.
 @constant WKNavigationActionPolicyDownload    Turn the navigation into a download.
 */
typedef NS_ENUM(NSInteger, WKBackgroundFetchChange) {
    WKBackgroundFetchChangeAddition,
    WKBackgroundFetchChangeRemoval,
    WKBackgroundFetchChangeUpdate,
} WK_API_AVAILABLE(macos(14.0), ios(17.0));

typedef NS_ENUM(NSInteger, WKWindowProxyProperty) {
    WKWindowProxyPropertyInitialOpen,
    WKWindowProxyPropertyPostMessage,
    WKWindowProxyPropertyClosed,
    WKWindowProxyPropertyOther,
} WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

#define HAVE_WK_WINDOW_PROXY_PROPERTY 1

WK_API_AVAILABLE(macos(10.15), ios(13.0))
@protocol _WKWebsiteDataStoreDelegate <NSObject>

@optional
- (void)requestStorageSpace:(NSURL *)mainFrameURL frameOrigin:(NSURL *)frameURL quota:(NSUInteger)quota currentSize:(NSUInteger)currentSize spaceRequired:(NSUInteger)spaceRequired decisionHandler:(void (^)(unsigned long long quota))decisionHandler;
- (void)didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential))completionHandler;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore openWindow:(NSURL *)url fromServiceWorkerOrigin:(WKSecurityOrigin *)serviceWorkerOrigin completionHandler:(void (^)(WKWebView *newWebView))completionHandler;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore reportServiceWorkerConsoleMessage:(NSString *)message;
- (NSDictionary<NSString *, NSNumber *> *)notificationPermissionsForWebsiteDataStore:(WKWebsiteDataStore *)dataStore;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore showNotification:(_WKNotificationData *)notificationData;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore workerOrigin:(WKSecurityOrigin *)workerOrigin updatedAppBadge:(NSNumber *)badge;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore getDisplayedNotificationsForWorkerOrigin:(WKSecurityOrigin *)workerOrigin completionHandler:(void (^)(NSArray<NSDictionary *> *))completionHandler;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore navigateToNotificationActionURL:(NSURL *)url;
- (void)requestBackgroundFetchPermission:(NSURL *)mainFrameURL frameOrigin:(NSURL *)frameURL  decisionHandler:(void (^)(bool isGranted))decisionHandler;
- (void)notifyBackgroundFetchChange:(NSString *)backgroundFetchIdentifier change:(WKBackgroundFetchChange)change;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore domain:(NSString *)registrableDomain didOpenDomainViaWindowOpen:(NSString *)openedRegistrableDomain withProperty:(WKWindowProxyProperty)property directly:(BOOL)directly;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore didAllowPrivateTokenUsageByThirdPartyForTesting:(BOOL)wasAllowed forResourceURL:(NSURL *)resourceURL;
- (void)websiteDataStore:(WKWebsiteDataStore *)dataStore domain:(NSString *)registrableDomain didExceedMemoryFootprintThreshold:(size_t)footprint withPageCount:(NSUInteger)pageCount processLifetime:(NSTimeInterval)processLifetime inForeground:(BOOL)inForeground wasPrivateRelayed:(BOOL)wasPrivateRelayed canSuspend:(BOOL)canSuspend;
- (void)webCryptoMasterKey:(void (^)(NSData *))completionHandler WK_API_AVAILABLE(macos(15.0), ios(18.0));
@end
