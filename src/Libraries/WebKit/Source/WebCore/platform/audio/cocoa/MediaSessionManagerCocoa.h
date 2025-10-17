/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#if PLATFORM(COCOA)

#include "AudioHardwareListener.h"
#include "AudioSession.h"
#include "NowPlayingManager.h"
#include "PlatformMediaSessionManager.h"
#include "RemoteCommandListener.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct NowPlayingInfo;

enum class MediaPlayerPitchCorrectionAlgorithm : uint8_t;

class MediaSessionManagerCocoa
    : public PlatformMediaSessionManager
    , private NowPlayingManagerClient
    , private AudioHardwareListener::Client {
    WTF_MAKE_TZONE_ALLOCATED(MediaSessionManagerCocoa);
public:
    MediaSessionManagerCocoa();
    
    void updateSessionState() final;
    void beginInterruption(PlatformMediaSession::InterruptionType) final;

    bool hasActiveNowPlayingSession() const final { return m_nowPlayingActive; }
    String lastUpdatedNowPlayingTitle() const final { return m_lastUpdatedNowPlayingTitle; }
    double lastUpdatedNowPlayingDuration() const final { return m_lastUpdatedNowPlayingDuration; }
    double lastUpdatedNowPlayingElapsedTime() const final { return m_lastUpdatedNowPlayingElapsedTime; }
    std::optional<MediaUniqueIdentifier> lastUpdatedNowPlayingInfoUniqueIdentifier() const final { return m_lastUpdatedNowPlayingInfoUniqueIdentifier; }
    bool registeredAsNowPlayingApplication() const final { return m_registeredAsNowPlayingApplication; }
    bool haveEverRegisteredAsNowPlayingApplication() const final { return m_haveEverRegisteredAsNowPlayingApplication; }

    void prepareToSendUserMediaPermissionRequestForPage(Page&) final;

    std::optional<NowPlayingInfo> nowPlayingInfo() const final { return m_nowPlayingInfo; }
    static WEBCORE_EXPORT void clearNowPlayingInfo();
    static WEBCORE_EXPORT void setNowPlayingInfo(bool setAsNowPlayingApplication, bool shouldUpdateNowPlayingSuppression, const NowPlayingInfo&);

    static WEBCORE_EXPORT void updateMediaUsage(PlatformMediaSession&);

    static void ensureCodecsRegistered();

    static WEBCORE_EXPORT void setShouldUseModernAVContentKeySession(bool);
    static WEBCORE_EXPORT bool shouldUseModernAVContentKeySession();

    static String audioTimePitchAlgorithmForMediaPlayerPitchCorrectionAlgorithm(MediaPlayerPitchCorrectionAlgorithm, bool preservesPitch, double rate);

protected:
    void scheduleSessionStatusUpdate() final;
    void updateNowPlayingInfo();
    void updateActiveNowPlayingSession(CheckedPtr<PlatformMediaSession>);

    void removeSession(PlatformMediaSession&) final;
    void addSession(PlatformMediaSession&) final;
    void setCurrentSession(PlatformMediaSession&) final;

    bool sessionWillBeginPlayback(PlatformMediaSession&) override;
    void sessionWillEndPlayback(PlatformMediaSession&, DelayCallingUpdateNowPlaying) override;
    void sessionDidEndRemoteScrubbing(PlatformMediaSession&) final;
    void clientCharacteristicsChanged(PlatformMediaSession&, bool) final;
    void sessionCanProduceAudioChanged() final;

    virtual void providePresentingApplicationPIDIfNecessary(ProcessID) { }

    WeakPtr<PlatformMediaSession> nowPlayingEligibleSession();

    void addSupportedCommand(PlatformMediaSession::RemoteControlCommandType) final;
    void removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType) final;
    RemoteCommandListener::RemoteCommandsSet supportedCommands() const final;

    void resetHaveEverRegisteredAsNowPlayingApplicationForTesting() final { m_haveEverRegisteredAsNowPlayingApplication = false; };
    void resetSessionState() final;

private:
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "MediaSessionManagerCocoa"_s; }
#endif

    // NowPlayingManagerClient
    void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType type, const PlatformMediaSession::RemoteCommandArgument& argument) final { processDidReceiveRemoteControlCommand(type, argument); }

    // AudioHardwareListenerClient
    void audioHardwareDidBecomeActive() final { }
    void audioHardwareDidBecomeInactive() final { }
    void audioOutputDeviceChanged() final;

    void possiblyChangeAudioCategory();

    std::optional<bool> supportsSpatialAudioPlaybackForConfiguration(const MediaConfiguration&) final;

#if USE(NOW_PLAYING_ACTIVITY_SUPPRESSION)
    static void updateNowPlayingSuppression(const NowPlayingInfo*);
#endif

    bool m_nowPlayingActive { false };
    bool m_registeredAsNowPlayingApplication { false };
    bool m_haveEverRegisteredAsNowPlayingApplication { false };

    // For testing purposes only.
    String m_lastUpdatedNowPlayingTitle;
    double m_lastUpdatedNowPlayingDuration { NAN };
    double m_lastUpdatedNowPlayingElapsedTime { NAN };
    Markable<MediaUniqueIdentifier> m_lastUpdatedNowPlayingInfoUniqueIdentifier;
    std::optional<NowPlayingInfo> m_nowPlayingInfo;

    const std::unique_ptr<NowPlayingManager> m_nowPlayingManager;
    RefPtr<AudioHardwareListener> m_audioHardwareListener;

    AudioHardwareListener::BufferSizeRange m_supportedAudioHardwareBufferSizes;
    size_t m_defaultBufferSize;

    RunLoop::Timer m_delayCategoryChangeTimer;
    AudioSession::CategoryType m_previousCategory { AudioSession::CategoryType::None };
    bool m_previousHadAudibleAudioOrVideoMediaType { false };
};

} // namespace WebCore

#endif // PLATFORM(COCOA)
