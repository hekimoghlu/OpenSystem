/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

#if PLATFORM(IOS_FAMILY)

#include "AudioSession.h"
#include "MediaSessionHelperIOS.h"
#include "MediaSessionManagerCocoa.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebMediaSessionHelper;

#if defined(__OBJC__) && __OBJC__
extern NSString *WebUIApplicationWillResignActiveNotification;
extern NSString *WebUIApplicationWillEnterForegroundNotification;
extern NSString *WebUIApplicationDidBecomeActiveNotification;
extern NSString *WebUIApplicationDidEnterBackgroundNotification;
#endif

namespace WebCore {

class MediaSessionManageriOS
    : public MediaSessionManagerCocoa
    , public MediaSessionHelperClient
    , public AudioSessionInterruptionObserver {
    WTF_MAKE_TZONE_ALLOCATED(MediaSessionManageriOS);
public:
    virtual ~MediaSessionManageriOS();

    bool hasWirelessTargetsAvailable() override;
    bool isMonitoringWirelessTargets() const override;

    USING_CAN_MAKE_WEAKPTR(MediaSessionHelperClient);

private:
    friend class PlatformMediaSessionManager;

    MediaSessionManageriOS();

#if !PLATFORM(MACCATALYST)
    void resetRestrictions() final;
#endif

    void configureWirelessTargetMonitoring() final;
    void providePresentingApplicationPIDIfNecessary(ProcessID) final;
    void updatePresentingApplicationPIDIfNecessary(ProcessID) final;
    bool sessionWillBeginPlayback(PlatformMediaSession&) final;
    void sessionWillEndPlayback(PlatformMediaSession&, DelayCallingUpdateNowPlaying) final;

    // AudioSessionInterruptionObserver
    void beginAudioSessionInterruption() final { beginInterruption(PlatformMediaSession::InterruptionType::SystemInterruption); }
    void endAudioSessionInterruption(AudioSession::MayResume mayResume) final { endInterruption(mayResume == AudioSession::MayResume::Yes ? PlatformMediaSession::EndInterruptionFlags::MayResumePlaying : PlatformMediaSession::EndInterruptionFlags::NoFlags); }

    // MediaSessionHelperClient
    void applicationWillEnterForeground(SuspendedUnderLock) final;
    void applicationDidEnterBackground(SuspendedUnderLock) final;
    void applicationWillBecomeInactive() final;
    void applicationDidBecomeActive() final;
    void externalOutputDeviceAvailableDidChange(HasAvailableTargets) final;
    void activeAudioRouteDidChange(ShouldPause) final;
    void activeVideoRouteDidChange(SupportsAirPlayVideo, Ref<MediaPlaybackTarget>&&) final;
    void isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit) final;
    void activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback) final;
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "MediaSessionManageriOS"_s; }
#endif

#if !PLATFORM(WATCHOS)
    RefPtr<MediaPlaybackTarget> m_playbackTarget;
    bool m_playbackTargetSupportsAirPlayVideo { false };
#endif

    bool m_isMonitoringWirelessRoutes { false };
    bool m_havePresentedApplicationPID { false };
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
