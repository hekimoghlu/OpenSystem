/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#include "config.h"
#include "WebPreferencesDefaultValues.h"

#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/NumberOfCores.h>
#include <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#if PLATFORM(IOS_FAMILY)
#import <pal/system/ios/UserInterfaceIdiom.h>
#endif
#endif

#if ENABLE(MEDIA_SESSION_COORDINATOR)
#import "WebProcess.h"
#import <wtf/cocoa/Entitlements.h>
#endif

#if USE(LIBWEBRTC)
#include <WebCore/LibWebRTCProvider.h>
#endif

#if USE(APPLE_INTERNAL_SDK)
#include <WebKitAdditions/WebPreferencesDefaultValuesAdditions.h>
#endif

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/DeprecatedGlobalSettingsAdditions.cpp>)
#include <WebKitAdditions/DeprecatedGlobalSettingsAdditions.cpp>
#endif

namespace WebKit {

#if PLATFORM(IOS_FAMILY)

bool defaultPassiveTouchListenersAsDefaultOnDocument()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DefaultsToPassiveTouchListenersOnDocument);
    return result;
}

bool defaultCSSOMViewScrollingAPIEnabled()
{
    static bool result = WTF::IOSApplication::isIMDb() && !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::NoIMDbCSSOMViewScrollingQuirk);
    return !result;
}

bool defaultShouldPrintBackgrounds()
{
    static bool result = !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DefaultsToExcludingBackgroundsWhenPrinting);
    return result;
}

bool defaultAlternateFormControlDesignEnabled()
{
    return PAL::currentUserInterfaceIdiomIsVision();
}

#endif

#if ENABLE(FULLSCREEN_API)

bool defaultVideoFullscreenRequiresElementFullscreen()
{
#if USE(APPLE_INTERNAL_SDK)
    if (videoFullscreenRequiresElementFullscreenFromAdditions())
        return true;
#endif

#if PLATFORM(IOS_FAMILY)
    if (PAL::currentUserInterfaceIdiomIsVision())
        return true;
#endif

    return false;
}

#endif

#if PLATFORM(MAC)

bool defaultPassiveWheelListenersAsDefaultOnDocument()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DefaultsToPassiveWheelListenersOnDocument);
    return result;
}

bool defaultWheelEventGesturesBecomeNonBlocking()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::AllowsWheelEventGesturesToBecomeNonBlocking);
    return result;
}

#endif

#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)

bool defaultDisallowSyncXHRDuringPageDismissalEnabled()
{
#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
    if (CFPreferencesGetAppBooleanValue(CFSTR("allowDeprecatedSynchronousXMLHttpRequestDuringUnload"), CFSTR("com.apple.WebKit"), nullptr)) {
        WTFLogAlways("Allowing synchronous XHR during page unload due to managed preference");
        return false;
    }
#elif PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST) && !PLATFORM(WATCHOS)
    if (allowsDeprecatedSynchronousXMLHttpRequestDuringUnload()) {
        WTFLogAlways("Allowing synchronous XHR during page unload due to managed preference");
        return false;
    }
#endif
    return true;
}

#endif

#if PLATFORM(MAC)

bool defaultAppleMailPaginationQuirkEnabled()
{
    return WTF::MacApplication::isAppleMail();
}

#endif

#if ENABLE(MEDIA_STREAM)

bool defaultCaptureAudioInGPUProcessEnabled()
{
#if HAVE(REQUIRE_MICROPHONE_CAPTURE_IN_UIPROCESS)
    // Newer versions can capture microphone in GPUProcess.
    if (!WTF::MacApplication::isSafari())
        return false;
#endif

#if ENABLE(GPU_PROCESS_BY_DEFAULT)
    return true;
#else
    return false;
#endif
}

bool defaultCaptureAudioInUIProcessEnabled()
{
#if PLATFORM(MAC)
    return !defaultCaptureAudioInGPUProcessEnabled();
#endif

    return false;
}

bool defaultManageCaptureStatusBarInGPUProcessEnabled()
{
#if PLATFORM(IOS_FAMILY)
    // FIXME: Enable by default for all applications.
    return !WTF::IOSApplication::isMobileSafari() && !WTF::IOSApplication::isSafariViewService();
#else
    return false;
#endif
}

#endif // ENABLE(MEDIA_STREAM)

#if ENABLE(MEDIA_SOURCE)
bool defaultManagedMediaSourceEnabled()
{
#if PLATFORM(COCOA)
    return true;
#else
    return false;
#endif
}
#endif

#if ENABLE(MEDIA_SOURCE) && ENABLE(WIRELESS_PLAYBACK_TARGET)
bool defaultManagedMediaSourceNeedsAirPlay()
{
#if PLATFORM(IOS_FAMILY) || PLATFORM(MAC)
    return true;
#else
    return false;
#endif
}
#endif

#if ENABLE(MEDIA_SESSION_COORDINATOR)
bool defaultMediaSessionCoordinatorEnabled()
{
    static dispatch_once_t onceToken;
    static bool enabled { false };
    dispatch_once(&onceToken, ^{
        if (isInWebProcess())
            enabled = WebProcess::singleton().parentProcessHasEntitlement("com.apple.developer.group-session.urlactivity"_s);
        else
            enabled = WTF::processHasEntitlement("com.apple.developer.group-session.urlactivity"_s);
    });
    return enabled;
}
#endif

bool defaultRunningBoardThrottlingEnabled()
{
#if PLATFORM(MAC)
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::RunningBoardThrottling);
    return newSDK;
#else
    return false;
#endif
}

bool defaultShouldDropNearSuspendedAssertionAfterDelay()
{
#if PLATFORM(COCOA)
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::FullySuspendsBackgroundContent);
    return newSDK;
#else
    return false;
#endif
}

bool defaultShouldTakeNearSuspendedAssertion()
{
#if PLATFORM(COCOA)
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::FullySuspendsBackgroundContentImmediately);
    return !newSDK;
#else
    return true;
#endif
}

bool defaultLinearMediaPlayerEnabled()
{
#if ENABLE(LINEAR_MEDIA_PLAYER)
    return PAL::currentUserInterfaceIdiomIsVision();
#else
    return false;
#endif
}

bool defaultLiveRangeSelectionEnabled()
{
#if PLATFORM(IOS_FAMILY)
    static bool enableForAllApps = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::LiveRangeSelectionEnabledForAllApps);
    if (!enableForAllApps && WTF::IOSApplication::isGmail())
        return false;
#endif
    return true;
}

bool defaultShowModalDialogEnabled()
{
#if PLATFORM(COCOA)
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::NoShowModalDialog);
    return !newSDK;
#else
    return false;
#endif
}

#if ENABLE(GAMEPAD)
bool defaultGamepadVibrationActuatorEnabled()
{
#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    return true;
#else
    return false;
#endif
}
#endif

bool defaultShouldEnableScreenOrientationAPI()
{
#if PLATFORM(MAC)
    return true;
#elif PLATFORM(IOS_FAMILY)
    static bool shouldEnableScreenOrientationAPI = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ScreenOrientationAPIEnabled) || WTF::IOSApplication::isHoYoLAB();
    return shouldEnableScreenOrientationAPI;
#else
    return false;
#endif
}

#if USE(LIBWEBRTC)
bool defaultPeerConnectionEnabledAvailable()
{
    // This helper function avoid an expensive header include in WebPreferences.h
    return WebCore::WebRTCProvider::webRTCAvailable();
}
#endif

bool defaultPopoverAttributeEnabled()
{
#if PLATFORM(COCOA)
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::PopoverAttributeEnabled);
    return newSDK;
#else
    return true;
#endif
}

bool defaultUseGPUProcessForDOMRenderingEnabled()
{
#if ENABLE(GPU_PROCESS_BY_DEFAULT) && ENABLE(GPU_PROCESS_DOM_RENDERING_BY_DEFAULT)
#if PLATFORM(MAC)
    static bool haveSufficientCores = WTF::numberOfPhysicalProcessorCores() >= 4;
    return haveSufficientCores;
#else
    return true;
#endif
#endif

#if USE(GRAPHICS_LAYER_WC)
    return true;
#endif

    return false;
}

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
bool defaultBuiltInNotificationsEnabled()
{
#if defined(DEPRECATED_GLOBAL_SETTINGS_BUILT_IN_NOTIFICATIONS_ENABLED_ADDITIONS)
    DEPRECATED_GLOBAL_SETTINGS_BUILT_IN_NOTIFICATIONS_ENABLED_ADDITIONS;
#endif

#if defined(WEB_PREFERENCES_BUILT_IN_NOTIFICATIONS_ENABLED_ADDITIONS)
    WEB_PREFERENCES_BUILT_IN_NOTIFICATIONS_ENABLED_ADDITIONS;
#endif

    return false;
}
#endif

bool defaultRequiresPageVisibilityForVideoToBeNowPlaying()
{
#if USE(APPLE_INTERNAL_SDK)
    if (requiresPageVisibilityForVideoToBeNowPlayingFromAdditions())
        return true;
#endif

    return false;
}

bool defaultCookieStoreAPIEnabled()
{
#if ENABLE(COOKIE_STORE_API_BY_DEFAULT)
    return true;
#else
    return false;
#endif
}

} // namespace WebKit
