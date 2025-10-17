/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#if USE(GLIB) && ENABLE(MEDIA_SESSION)

#include "MediaSessionIdentifier.h"
#include "NowPlayingManager.h"
#include "PlatformMediaSessionManager.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>

namespace WebCore {

struct NowPlayingInfo;
class MediaSessionGLib;

class MediaSessionManagerGLib
    : public PlatformMediaSessionManager
    , private NowPlayingManagerClient {
    WTF_MAKE_TZONE_ALLOCATED(MediaSessionManagerGLib);
public:
    MediaSessionManagerGLib(GRefPtr<GDBusNodeInfo>&&);
    ~MediaSessionManagerGLib();

    void beginInterruption(PlatformMediaSession::InterruptionType) final;

    bool hasActiveNowPlayingSession() const final { return m_nowPlayingActive; }
    String lastUpdatedNowPlayingTitle() const final { return m_lastUpdatedNowPlayingTitle; }
    double lastUpdatedNowPlayingDuration() const final { return m_lastUpdatedNowPlayingDuration; }
    double lastUpdatedNowPlayingElapsedTime() const final { return m_lastUpdatedNowPlayingElapsedTime; }
    std::optional<MediaUniqueIdentifier> lastUpdatedNowPlayingInfoUniqueIdentifier() const final { return m_lastUpdatedNowPlayingInfoUniqueIdentifier; }
    bool registeredAsNowPlayingApplication() const final { return m_registeredAsNowPlayingApplication; }
    bool haveEverRegisteredAsNowPlayingApplication() const final { return m_haveEverRegisteredAsNowPlayingApplication; }

    void dispatch(PlatformMediaSession::RemoteControlCommandType, PlatformMediaSession::RemoteCommandArgument);

    const GRefPtr<GDBusNodeInfo>& mprisInterface() const { return m_mprisInterface; }
    void setPrimarySessionIfNeeded(PlatformMediaSession&);
    void unregisterAllOtherSessions(PlatformMediaSession&);
    WeakPtr<PlatformMediaSession> nowPlayingEligibleSession();

    void setDBusNotificationsEnabled(bool dbusNotificationsEnabled) { m_dbusNotificationsEnabled = dbusNotificationsEnabled; }
    bool areDBusNotificationsEnabled() const { return m_dbusNotificationsEnabled; }

protected:
    void scheduleSessionStatusUpdate() final;
    void updateNowPlayingInfo();

    void removeSession(PlatformMediaSession&) final;
    void addSession(PlatformMediaSession&) final;
    void setCurrentSession(PlatformMediaSession&) final;

    bool sessionWillBeginPlayback(PlatformMediaSession&) override;
    void sessionWillEndPlayback(PlatformMediaSession&, DelayCallingUpdateNowPlaying) override;
    void sessionStateChanged(PlatformMediaSession&) override;
    void sessionDidEndRemoteScrubbing(PlatformMediaSession&) final;
    void clientCharacteristicsChanged(PlatformMediaSession&, bool) final;
    void sessionCanProduceAudioChanged() final;

    virtual void providePresentingApplicationPIDIfNecessary() { }

    void addSupportedCommand(PlatformMediaSession::RemoteControlCommandType) final;
    void removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType) final;
    RemoteCommandListener::RemoteCommandsSet supportedCommands() const final;

    void resetHaveEverRegisteredAsNowPlayingApplicationForTesting() final { m_haveEverRegisteredAsNowPlayingApplication = false; };

private:
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "MediaSessionManagerGLib"_s; }
#endif

    // NowPlayingManagerClient
    void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType type, const PlatformMediaSession::RemoteCommandArgument& argument) final { processDidReceiveRemoteControlCommand(type, argument); }

    bool m_isSeeking { false };
    GRefPtr<GDBusNodeInfo> m_mprisInterface;

    bool m_nowPlayingActive { false };
    bool m_registeredAsNowPlayingApplication { false };
    bool m_haveEverRegisteredAsNowPlayingApplication { false };

    // For testing purposes only.
    String m_lastUpdatedNowPlayingTitle;
    double m_lastUpdatedNowPlayingDuration { NAN };
    double m_lastUpdatedNowPlayingElapsedTime { NAN };
    Markable<MediaUniqueIdentifier> m_lastUpdatedNowPlayingInfoUniqueIdentifier;

    const std::unique_ptr<NowPlayingManager> m_nowPlayingManager;
    UncheckedKeyHashMap<MediaSessionIdentifier, std::unique_ptr<MediaSessionGLib>> m_sessions;

    bool m_dbusNotificationsEnabled { true };
};

} // namespace WebCore

#endif // USE(GLIB) && ENABLE(MEDIA_SESSION)
