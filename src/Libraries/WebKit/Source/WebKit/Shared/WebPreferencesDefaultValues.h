/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#pragma once

#include <wtf/Forward.h>

#if PLATFORM(IOS_FAMILY)
#define EXPERIMENTAL_FULLSCREEN_API_HIDDEN false
#else
#define EXPERIMENTAL_FULLSCREEN_API_HIDDEN true
#endif

// FIXME: https://bugs.webkit.org/show_bug.cgi?id=269475 - this should not be needed
#if defined(ENABLE_WEBGPU_BY_DEFAULT) && ENABLE_WEBGPU_BY_DEFAULT
#define Webgpu_feature_status Stable
#else
#define Webgpu_feature_status Preview
#endif

#if defined(ENABLE_WEBGPU_BY_DEFAULT) && ENABLE_WEBGPU_BY_DEFAULT && defined(ENABLE_WEBGPU_HDR_BY_DEFAULT) && ENABLE_WEBGPU_HDR_BY_DEFAULT
#define Webgpuhdr_feature_status Stable
#else
#define Webgpuhdr_feature_status Preview
#endif

#if defined(ENABLE_WEBXR_WEBGPU_BY_DEFAULT) && ENABLE_WEBXR_WEBGPU_BY_DEFAULT && PLATFORM(VISION)
#define Webxr_layers_feature_status Stable
#else
#define Webxr_layers_feature_status Unstable
#endif

#if defined(ENABLE_WEBXR_WEBGPU_BY_DEFAULT) && ENABLE_WEBXR_WEBGPU_BY_DEFAULT && PLATFORM(VISION)
#define Webgpu_webxr_feature_status Stable
#else
#define Webgpu_webxr_feature_status Unstable
#endif

#if defined(ENABLE_UNIFIED_PDF_BY_DEFAULT) && ENABLE_UNIFIED_PDF_BY_DEFAULT
#define Unifiedpdf_feature_status Mature
#elif defined(ENABLE_UNIFIED_PDF_AS_PREVIEW) && ENABLE_UNIFIED_PDF_AS_PREVIEW
#define Unifiedpdf_feature_status Preview
#else
#define Unifiedpdf_feature_status Internal
#endif

#if defined(ENABLE_UNPREFIXED_BACKDROP_FILTER) && ENABLE_UNPREFIXED_BACKDROP_FILTER
#define Backdropfilter_feature_status Stable
#else
#define Backdropfilter_feature_status Testable
#endif

namespace WebKit {

#if PLATFORM(IOS_FAMILY)
bool defaultPassiveTouchListenersAsDefaultOnDocument();
bool defaultCSSOMViewScrollingAPIEnabled();
bool defaultShouldPrintBackgrounds();
bool defaultAlternateFormControlDesignEnabled();
bool defaultUseAsyncUIKitInteractions();
bool defaultWriteRichTextDataWhenCopyingOrDragging();
#if ENABLE(TEXT_AUTOSIZING)
bool defaultTextAutosizingUsesIdempotentMode();
#endif
#endif

#if ENABLE(FULLSCREEN_API)
bool defaultVideoFullscreenRequiresElementFullscreen();
#endif

#if PLATFORM(MAC)
bool defaultScrollAnimatorEnabled();
bool defaultPassiveWheelListenersAsDefaultOnDocument();
bool defaultWheelEventGesturesBecomeNonBlocking();
#endif

#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)
bool defaultDisallowSyncXHRDuringPageDismissalEnabled();
#endif

#if PLATFORM(MAC)
bool defaultAppleMailPaginationQuirkEnabled();
#endif

#if !PLATFORM(MACCATALYST) && !PLATFORM(WATCHOS)
bool allowsDeprecatedSynchronousXMLHttpRequestDuringUnload();
#endif

#if ENABLE(MEDIA_STREAM)
bool defaultCaptureAudioInGPUProcessEnabled();
bool defaultCaptureAudioInUIProcessEnabled();
bool defaultManageCaptureStatusBarInGPUProcessEnabled();
#endif

#if ENABLE(MEDIA_SOURCE) && PLATFORM(IOS_FAMILY)
bool defaultMediaSourceEnabled();
#endif

#if ENABLE(MEDIA_SOURCE)
bool defaultManagedMediaSourceEnabled();
#if ENABLE(WIRELESS_PLAYBACK_TARGET)
bool defaultManagedMediaSourceNeedsAirPlay();
#endif
#endif

#if ENABLE(MEDIA_SESSION_COORDINATOR)
bool defaultMediaSessionCoordinatorEnabled();
#endif

#if ENABLE(IMAGE_ANALYSIS)
bool defaultTextRecognitionInVideosEnabled();
bool defaultVisualTranslationEnabled();
bool defaultRemoveBackgroundEnabled();
#endif

#if ENABLE(GAMEPAD)
bool defaultGamepadVibrationActuatorEnabled();
#endif

#if PLATFORM(IOS_FAMILY)
bool defaultAutomaticLiveResizeEnabled();
bool defaultVisuallyContiguousBidiTextSelectionEnabled();
bool defaultBidiContentAwarePasteEnabled();
#endif

bool defaultRunningBoardThrottlingEnabled();
bool defaultShouldDropNearSuspendedAssertionAfterDelay();
bool defaultShouldTakeNearSuspendedAssertion();
bool defaultShowModalDialogEnabled();
bool defaultLinearMediaPlayerEnabled();
bool defaultLiveRangeSelectionEnabled();

bool defaultShouldEnableScreenOrientationAPI();
bool defaultPopoverAttributeEnabled();
bool defaultUseGPUProcessForDOMRenderingEnabled();

#if HAVE(SC_CONTENT_SHARING_PICKER)
bool defaultUseSCContentSharingPicker();
#endif

#if USE(LIBWEBRTC)
bool defaultPeerConnectionEnabledAvailable();
#endif

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
bool defaultBuiltInNotificationsEnabled();
#endif

bool defaultRequiresPageVisibilityForVideoToBeNowPlaying();

bool defaultCookieStoreAPIEnabled();

} // namespace WebKit
