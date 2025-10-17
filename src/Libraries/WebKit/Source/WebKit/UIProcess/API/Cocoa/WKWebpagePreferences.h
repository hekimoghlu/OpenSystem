/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

/*! @enum WKContentMode
 @abstract A content mode represents the type of content to load, as well as
 additional layout and rendering adaptations that are applied as a result of
 loading the content
 @constant WKContentModeRecommended  The recommended content mode for the current platform
 @constant WKContentModeMobile       Represents content targeting mobile browsers
 @constant WKContentModeDesktop      Represents content targeting desktop browsers
 @discussion WKContentModeRecommended behaves like WKContentModeMobile on iPhone and iPad mini
 and WKContentModeDesktop on other iPad models as well as Mac.
 */
typedef NS_ENUM(NSInteger, WKContentMode) {
    WKContentModeRecommended,
    WKContentModeMobile,
    WKContentModeDesktop
} WK_API_AVAILABLE(ios(13.0));

/*! @enum WKWebpagePreferencesUpgradeToHTTPSPolicy
 @abstract A secure navigation policy represents whether or not there is a
 preference for loading a webpage with https, and how failures should be
 handled.
 @constant WKWebpagePreferencesUpgradeToHTTPSPolicyKeepAsRequested             Maintains the current behavior without preferring https
 @constant WKWebpagePreferencesUpgradeToHTTPSPolicyAutomaticFallbackToHTTP     Upgrades http requests to https, and re-attempts the request with http on failure
 @constant WKWebpagePreferencesUpgradeToHTTPSPolicyUserMediatedFallbackToHTTP  Upgrades http requests to https, and shows a warning page on failure
 @constant WKWebpagePreferencesUpgradeToHTTPSPolicyErrorOnFailure              Upgrades http requests to https, and returns an error on failure
 */
typedef NS_ENUM(NSInteger, WKWebpagePreferencesUpgradeToHTTPSPolicy) {
    WKWebpagePreferencesUpgradeToHTTPSPolicyKeepAsRequested,
    WKWebpagePreferencesUpgradeToHTTPSPolicyAutomaticFallbackToHTTP,
    WKWebpagePreferencesUpgradeToHTTPSPolicyUserMediatedFallbackToHTTP,
    WKWebpagePreferencesUpgradeToHTTPSPolicyErrorOnFailure
} NS_SWIFT_NAME(WKWebpagePreferences.UpgradeToHTTPSPolicy) WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

/*! A WKWebpagePreferences object is a collection of properties that
 determine the preferences to use when loading and rendering a page.
 @discussion Contains properties used to determine webpage preferences.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.15), ios(13.0))
@interface WKWebpagePreferences : NSObject

/*! @abstract A WKContentMode indicating the content mode to prefer
 when loading and rendering a webpage.
 @discussion The default value is WKContentModeRecommended. The stated
 preference is ignored on subframe navigation
 */
@property (nonatomic) WKContentMode preferredContentMode WK_API_AVAILABLE(ios(13.0));

/* @abstract A Boolean value indicating whether JavaScript from web content is enabled
 @discussion If this value is set to NO then JavaScript referenced by the web content will not execute.
 This includes JavaScript found in inline <script> elements, referenced by external JavaScript resources,
 "javascript:" URLs, and all other forms.

 Even if this value is set to NO your application can still execute JavaScript using:
 - [WKWebView evaluteJavaScript:completionHandler:]
 - [WKWebView evaluteJavaScript:inContentWorld:completionHandler:]
 - [WKWebView callAsyncJavaScript:arguments:inContentWorld:completionHandler:]
 - WKUserScripts

 The default value is YES.
*/
@property (nonatomic) BOOL allowsContentJavaScript WK_API_AVAILABLE(macos(11.0), ios(14.0));

/*! @abstract A boolean indicating whether lockdown mode is enabled.
 @discussion This mode trades off performance and compatibility in favor of security.
 The default value depends on the system setting.
 */
@property (nonatomic, getter=isLockdownModeEnabled) BOOL lockdownModeEnabled WK_API_AVAILABLE(macos(13.0), ios(16.0));

/*! @abstract A WKWebpagePreferencesUpgradeToHTTPSPolicy indicating the desired mode
 used when performing a top-level navigation to a webpage.
 @discussion The default value is WKWebpagePreferencesUpgradeToHTTPSPolicyKeepAsRequested.
 The stated preference is ignored on subframe navigation, and it may be ignored based on
 system configuration. The upgradeKnownHostsToHTTPS property on WKWebViewConfiguration
 supercedes this policy for known hosts.
 */
@property (nonatomic) WKWebpagePreferencesUpgradeToHTTPSPolicy preferredHTTPSNavigationPolicy WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

@end
