/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#import "config.h"
#import "MediaSessionManagerIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "Logging.h"
#import "MediaConfiguration.h"
#import "MediaPlaybackTargetCocoa.h"
#import "MediaPlayer.h"
#import "PlatformMediaSession.h"
#import "SystemMemory.h"
#import "WebCoreThreadRun.h"
#import <wtf/MainThread.h>
#import <wtf/RAMSize.h>
#import <wtf/RetainPtr.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/TZoneMalloc.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaSessionManageriOS);

std::unique_ptr<PlatformMediaSessionManager> PlatformMediaSessionManager::create()
{
    auto manager = std::unique_ptr<MediaSessionManageriOS>(new MediaSessionManageriOS);
    MediaSessionHelper::sharedHelper().addClient(*manager);
    return manager;
}

MediaSessionManageriOS::MediaSessionManageriOS()
    : MediaSessionManagerCocoa()
{
    AudioSession::sharedSession().addInterruptionObserver(*this);
}

MediaSessionManageriOS::~MediaSessionManageriOS()
{
    if (m_isMonitoringWirelessRoutes)
        MediaSessionHelper::sharedHelper().stopMonitoringWirelessRoutes();
    MediaSessionHelper::sharedHelper().removeClient(*this);
    AudioSession::sharedSession().removeInterruptionObserver(*this);
}

#if !PLATFORM(MACCATALYST)
void MediaSessionManageriOS::resetRestrictions()
{
    static const size_t systemMemoryRequiredForVideoInBackgroundTabs = 1024 * 1024 * 1024;

    ALWAYS_LOG(LOGIDENTIFIER);

    MediaSessionManagerCocoa::resetRestrictions();

    if (ramSize() < systemMemoryRequiredForVideoInBackgroundTabs) {
        ALWAYS_LOG(LOGIDENTIFIER, "restricting video in background tabs because system memory = ", ramSize());
        addRestriction(PlatformMediaSession::MediaType::Video, BackgroundTabPlaybackRestricted);
    }

    addRestriction(PlatformMediaSession::MediaType::Video, BackgroundProcessPlaybackRestricted);
    addRestriction(PlatformMediaSession::MediaType::WebAudio, BackgroundProcessPlaybackRestricted);
    addRestriction(PlatformMediaSession::MediaType::VideoAudio, ConcurrentPlaybackNotPermitted | BackgroundProcessPlaybackRestricted | SuspendedUnderLockPlaybackRestricted);
}
#endif

bool MediaSessionManageriOS::hasWirelessTargetsAvailable()
{
    return MediaSessionHelper::sharedHelper().isExternalOutputDeviceAvailable();
}

bool MediaSessionManageriOS::isMonitoringWirelessTargets() const
{
    return m_isMonitoringWirelessRoutes;
}

void MediaSessionManageriOS::configureWirelessTargetMonitoring()
{
#if !PLATFORM(WATCHOS)
    bool requiresMonitoring = anyOfSessions([] (auto& session) {
        return session.requiresPlaybackTargetRouteMonitoring();
    });

    if (requiresMonitoring == m_isMonitoringWirelessRoutes)
        return;

    m_isMonitoringWirelessRoutes = requiresMonitoring;

    ALWAYS_LOG(LOGIDENTIFIER, "requiresMonitoring = ", requiresMonitoring);

    if (requiresMonitoring)
        MediaSessionHelper::sharedHelper().startMonitoringWirelessRoutes();
    else
        MediaSessionHelper::sharedHelper().stopMonitoringWirelessRoutes();
#endif
}

void MediaSessionManageriOS::providePresentingApplicationPIDIfNecessary(ProcessID pid)
{
#if HAVE(MEDIAEXPERIENCE_AVSYSTEMCONTROLLER)
    if (m_havePresentedApplicationPID)
        return;
    m_havePresentedApplicationPID = true;
    MediaSessionHelper::sharedHelper().providePresentingApplicationPID(pid);
#else
    UNUSED_PARAM(pid);
#endif
}

void MediaSessionManageriOS::updatePresentingApplicationPIDIfNecessary(ProcessID pid)
{
#if HAVE(MEDIAEXPERIENCE_AVSYSTEMCONTROLLER)
    if (m_havePresentedApplicationPID)
        MediaSessionHelper::sharedHelper().providePresentingApplicationPID(pid, MediaSessionHelper::ShouldOverride::Yes);
#else
    UNUSED_PARAM(pid);
#endif
}

bool MediaSessionManageriOS::sessionWillBeginPlayback(PlatformMediaSession& session)
{
    if (!MediaSessionManagerCocoa::sessionWillBeginPlayback(session))
        return false;

#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(MACCATALYST) && !PLATFORM(WATCHOS)
    auto playbackTargetSupportsAirPlayVideo = MediaSessionHelper::sharedHelper().activeVideoRouteSupportsAirPlayVideo();
    ALWAYS_LOG(LOGIDENTIFIER, "Playback Target Supports AirPlay Video = ", playbackTargetSupportsAirPlayVideo);
    if (auto target = MediaSessionHelper::sharedHelper().playbackTarget(); target && playbackTargetSupportsAirPlayVideo)
        session.setPlaybackTarget(*target);
    session.setShouldPlayToPlaybackTarget(playbackTargetSupportsAirPlayVideo);
#endif

    providePresentingApplicationPIDIfNecessary(session.presentingApplicationPID());

    return true;
}

void MediaSessionManageriOS::sessionWillEndPlayback(PlatformMediaSession& session, DelayCallingUpdateNowPlaying delayCallingUpdateNowPlaying)
{
    MediaSessionManagerCocoa::sessionWillEndPlayback(session, delayCallingUpdateNowPlaying);

#if USE(AUDIO_SESSION)
    if (isApplicationInBackground() && !anyOfSessions([] (auto& session) { return session.state() == PlatformMediaSession::State::Playing; }))
        maybeDeactivateAudioSession();
#endif
}

void MediaSessionManageriOS::externalOutputDeviceAvailableDidChange(HasAvailableTargets haveTargets)
{
    ALWAYS_LOG(LOGIDENTIFIER, haveTargets);

    forEachSession([haveTargets] (auto& session) {
        session.externalOutputDeviceAvailableDidChange(haveTargets == HasAvailableTargets::Yes);
    });
}

void MediaSessionManageriOS::isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit playingToAutomotiveHeadUnit)
{
    setIsPlayingToAutomotiveHeadUnit(playingToAutomotiveHeadUnit == PlayingToAutomotiveHeadUnit::Yes);
}

void MediaSessionManageriOS::activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback supportsSpatialPlayback)
{
    setSupportsSpatialAudioPlayback(supportsSpatialPlayback == SupportsSpatialAudioPlayback::Yes);
}

std::optional<bool> MediaSessionManagerCocoa::supportsSpatialAudioPlaybackForConfiguration(const MediaConfiguration& configuration)
{
    ASSERT(configuration.audio);

    // Only multichannel audio can be spatially rendered on iOS.
    if (!configuration.audio || configuration.audio->channels.toDouble() <= 2)
        return { false };

    auto supportsSpatialAudioPlayback = this->supportsSpatialAudioPlayback();
    if (supportsSpatialAudioPlayback.has_value())
        return supportsSpatialAudioPlayback;

    MediaSessionHelper::sharedHelper().updateActiveAudioRouteSupportsSpatialPlayback();

    return this->supportsSpatialAudioPlayback();
}

void MediaSessionManageriOS::activeAudioRouteDidChange(ShouldPause shouldPause)
{
    ALWAYS_LOG(LOGIDENTIFIER, shouldPause);

    if (shouldPause != ShouldPause::Yes)
        return;

    forEachSession([](auto& session) {
        if (session.canProduceAudio() && !session.shouldOverridePauseDuringRouteChange())
            session.pauseSession();
    });
}

void MediaSessionManageriOS::activeVideoRouteDidChange(SupportsAirPlayVideo supportsAirPlayVideo, Ref<MediaPlaybackTarget>&& playbackTarget)
{
    ALWAYS_LOG(LOGIDENTIFIER, supportsAirPlayVideo);

#if !PLATFORM(WATCHOS)
    m_playbackTarget = playbackTarget.ptr();
    m_playbackTargetSupportsAirPlayVideo = supportsAirPlayVideo == SupportsAirPlayVideo::Yes;
#endif

    CheckedPtr nowPlayingSession = nowPlayingEligibleSession().get();
    if (!nowPlayingSession)
        return;

    nowPlayingSession->setPlaybackTarget(WTFMove(playbackTarget));
    nowPlayingSession->setShouldPlayToPlaybackTarget(supportsAirPlayVideo == SupportsAirPlayVideo::Yes);
}

void MediaSessionManageriOS::applicationWillEnterForeground(SuspendedUnderLock isSuspendedUnderLock)
{
    if (willIgnoreSystemInterruptions())
        return;

    MediaSessionManagerCocoa::applicationWillEnterForeground(isSuspendedUnderLock == SuspendedUnderLock::Yes);
}

void MediaSessionManageriOS::applicationDidBecomeActive()
{
    if (willIgnoreSystemInterruptions())
        return;

    MediaSessionManagerCocoa::applicationDidBecomeActive();
}

void MediaSessionManageriOS::applicationDidEnterBackground(SuspendedUnderLock isSuspendedUnderLock)
{
    if (willIgnoreSystemInterruptions())
        return;

    MediaSessionManagerCocoa::applicationDidEnterBackground(isSuspendedUnderLock == SuspendedUnderLock::Yes);
}

void MediaSessionManageriOS::applicationWillBecomeInactive()
{
    if (willIgnoreSystemInterruptions())
        return;

    MediaSessionManagerCocoa::applicationWillBecomeInactive();
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
