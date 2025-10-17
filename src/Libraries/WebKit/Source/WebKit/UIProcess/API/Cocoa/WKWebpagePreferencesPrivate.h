/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#import <WebKit/WKWebpagePreferences.h>

typedef NS_ENUM(NSInteger, _WKWebsiteAutoplayPolicy) {
    _WKWebsiteAutoplayPolicyDefault,
    _WKWebsiteAutoplayPolicyAllow,
    _WKWebsiteAutoplayPolicyAllowWithoutSound,
    _WKWebsiteAutoplayPolicyDeny
} WK_API_AVAILABLE(macos(10.13), ios(11.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsiteAutoplayQuirk) {
    _WKWebsiteAutoplayQuirkSynthesizedPauseEvents = 1 << 0,
    _WKWebsiteAutoplayQuirkInheritedUserGestures = 1 << 1,
    _WKWebsiteAutoplayQuirkArbitraryUserGestures = 1 << 2,
    _WKWebsiteAutoplayQuirkPerDocumentAutoplayBehavior = 1 << 3,
} WK_API_AVAILABLE(macos(10.13), ios(11.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsitePopUpPolicy) {
    _WKWebsitePopUpPolicyDefault,
    _WKWebsitePopUpPolicyAllow,
    _WKWebsitePopUpPolicyBlock,
} WK_API_AVAILABLE(macos(10.14), ios(12.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsiteDeviceOrientationAndMotionAccessPolicy) {
    _WKWebsiteDeviceOrientationAndMotionAccessPolicyAsk,
    _WKWebsiteDeviceOrientationAndMotionAccessPolicyGrant,
    _WKWebsiteDeviceOrientationAndMotionAccessPolicyDeny,
} WK_API_AVAILABLE(macos(10.14), ios(12.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsiteMouseEventPolicy) {
    // Indirect pointing devices will generate either touch or mouse events based on WebKit's default policy.
    _WKWebsiteMouseEventPolicyDefault,

#if TARGET_OS_IPHONE
    // Indirect pointing devices will always synthesize touch events and behave as if touch input is being used.
    _WKWebsiteMouseEventPolicySynthesizeTouchEvents,
#endif
} WK_API_AVAILABLE(macos(11.0), ios(14.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsiteModalContainerObservationPolicy) {
    _WKWebsiteModalContainerObservationPolicyDisabled,
    _WKWebsiteModalContainerObservationPolicyPrompt,
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

// Allow overriding the system color-scheme with a per-website preference.
typedef NS_OPTIONS(NSUInteger, _WKWebsiteColorSchemePreference) {
    _WKWebsiteColorSchemePreferenceNoPreference,
    _WKWebsiteColorSchemePreferenceLight,
    _WKWebsiteColorSchemePreferenceDark,
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

typedef NS_OPTIONS(NSUInteger, _WKWebsiteNetworkConnectionIntegrityPolicy) {
    _WKWebsiteNetworkConnectionIntegrityPolicyNone = 0,
    _WKWebsiteNetworkConnectionIntegrityPolicyEnabled = 1 << 0,
    _WKWebsiteNetworkConnectionIntegrityPolicyHTTPSFirst = 1 << 1,
    _WKWebsiteNetworkConnectionIntegrityPolicyHTTPSOnly = 1 << 2,
    _WKWebsiteNetworkConnectionIntegrityPolicyHTTPSOnlyExplicitlyBypassedForDomain = 1 << 3,
    _WKWebsiteNetworkConnectionIntegrityPolicyFailClosed = 1 << 4,
    _WKWebsiteNetworkConnectionIntegrityPolicyWebSearchContent WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 1 << 5,
    _WKWebsiteNetworkConnectionIntegrityPolicyEnhancedTelemetry WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 1 << 6,
    _WKWebsiteNetworkConnectionIntegrityPolicyRequestValidation WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 1 << 7,
    _WKWebsiteNetworkConnectionIntegrityPolicySanitizeLookalikeCharacters WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 1 << 8,
    _WKWebsiteNetworkConnectionIntegrityPolicyFailClosedForAllHosts WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA)) = 1 << 9,
} WK_API_AVAILABLE(macos(13.3), ios(16.4));

@class _WKCustomHeaderFields;
@class WKUserContentController;
@class WKWebsiteDataStore;

@interface WKWebpagePreferences (WKPrivate)

@property (nonatomic, setter=_setContentBlockersEnabled:) BOOL _contentBlockersEnabled;
@property (nonatomic, copy, setter=_setActiveContentRuleListActionPatterns:) NSDictionary<NSString *, NSSet<NSString *> *> *_activeContentRuleListActionPatterns WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, setter=_setAllowedAutoplayQuirks:) _WKWebsiteAutoplayQuirk _allowedAutoplayQuirks;
@property (nonatomic, setter=_setAutoplayPolicy:) _WKWebsiteAutoplayPolicy _autoplayPolicy;
@property (nonatomic, copy, setter=_setCustomHeaderFields:) NSArray<_WKCustomHeaderFields *> *_customHeaderFields;
@property (nonatomic, setter=_setPopUpPolicy:) _WKWebsitePopUpPolicy _popUpPolicy;
@property (nonatomic, strong, setter=_setWebsiteDataStore:) WKWebsiteDataStore *_websiteDataStore;
@property (nonatomic, strong, setter=_setUserContentController:) WKUserContentController *_userContentController WK_API_AVAILABLE(macos(11.0), ios(14.0));
@property (nonatomic, copy, setter=_setCustomUserAgent:) NSString *_customUserAgent;
@property (nonatomic, copy, setter=_setCustomUserAgentAsSiteSpecificQuirks:) NSString *_customUserAgentAsSiteSpecificQuirks;
@property (nonatomic, copy, setter=_setCustomNavigatorPlatform:) NSString *_customNavigatorPlatform;
@property (nonatomic, setter=_setDeviceOrientationAndMotionAccessPolicy:) _WKWebsiteDeviceOrientationAndMotionAccessPolicy _deviceOrientationAndMotionAccessPolicy;
@property (nonatomic, setter=_setAllowSiteSpecificQuirksToOverrideCompatibilityMode:) BOOL _allowSiteSpecificQuirksToOverrideCompatibilityMode;

@property (nonatomic, copy, setter=_setApplicationNameForUserAgentWithModernCompatibility:) NSString *_applicationNameForUserAgentWithModernCompatibility;

@property (nonatomic, setter=_setMouseEventPolicy:) _WKWebsiteMouseEventPolicy _mouseEventPolicy WK_API_AVAILABLE(macos(11.0), ios(14.0));
@property (nonatomic, setter=_setModalContainerObservationPolicy:) _WKWebsiteModalContainerObservationPolicy _modalContainerObservationPolicy WK_API_AVAILABLE(macos(13.0), ios(16.0));

@property (nonatomic, setter=_setCaptivePortalModeEnabled:) BOOL _captivePortalModeEnabled WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, setter=_setAllowPrivacyProxy:) BOOL _allowPrivacyProxy WK_API_AVAILABLE(macos(13.1), ios(16.2));

@property (nonatomic, setter=_setColorSchemePreference:) _WKWebsiteColorSchemePreference _colorSchemePreference;

@property (nonatomic, setter=_setNetworkConnectionIntegrityEnabled:) BOOL _networkConnectionIntegrityEnabled WK_API_AVAILABLE(macos(13.3), ios(16.4));
@property (nonatomic, setter=_setNetworkConnectionIntegrityPolicy:) _WKWebsiteNetworkConnectionIntegrityPolicy _networkConnectionIntegrityPolicy WK_API_AVAILABLE(macos(13.3), ios(16.4));

@property (nonatomic, copy, setter=_setVisibilityAdjustmentSelectorsIncludingShadowHosts:) NSArray<NSArray<NSSet<NSString *> *> *> *_visibilityAdjustmentSelectorsIncludingShadowHosts WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, copy, setter=_setVisibilityAdjustmentSelectors:) NSSet<NSString *> *_visibilityAdjustmentSelectors WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));

- (void)_setContentRuleListsEnabled:(BOOL)enabled exceptions:(NSSet<NSString *> *)exceptions WK_API_AVAILABLE(macos(14.0), ios(17.0));

@property (nonatomic, setter=_setPushAndNotificationAPIEnabled:) BOOL _pushAndNotificationAPIEnabled;

@end
