/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

#include "MediaUniqueIdentifier.h"
#include "NowPlayingMetadataObserver.h"
#include "PlatformMediaSession.h"
#include "RemoteCommandListener.h"
#include "Timer.h"
#include <wtf/AggregateLogger.h>
#include <wtf/CancellableTask.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Page;
class PlatformMediaSession;
struct MediaConfiguration;
struct NowPlayingInfo;
struct NowPlayingMetadata;

class PlatformMediaSessionManager
#if !RELEASE_LOG_DISABLED
    : private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(PlatformMediaSessionManager);
public:
    WEBCORE_EXPORT static PlatformMediaSessionManager* singletonIfExists();
    WEBCORE_EXPORT static PlatformMediaSessionManager& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    static void updateNowPlayingInfoIfNecessary();
    static void updateAudioSessionCategoryIfNecessary();

    WEBCORE_EXPORT static void setShouldDeactivateAudioSession(bool);
    WEBCORE_EXPORT static bool shouldDeactivateAudioSession();

    WEBCORE_EXPORT static void setAlternateWebMPlayerEnabled(bool);
    WEBCORE_EXPORT static bool alternateWebMPlayerEnabled();
    WEBCORE_EXPORT static void setUseSCContentSharingPicker(bool);
    WEBCORE_EXPORT static bool useSCContentSharingPicker();

#if ENABLE(VP9)
    WEBCORE_EXPORT static void setShouldEnableVP9Decoder(bool);
    WEBCORE_EXPORT static bool shouldEnableVP9Decoder();
    WEBCORE_EXPORT static void setSWVPDecodersAlwaysEnabled(bool);
    WEBCORE_EXPORT static bool swVPDecodersAlwaysEnabled();
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
    WEBCORE_EXPORT static bool mediaCapabilityGrantsEnabled();
    WEBCORE_EXPORT static void setMediaCapabilityGrantsEnabled(bool);
#endif

    virtual ~PlatformMediaSessionManager();

    virtual void scheduleSessionStatusUpdate() { }

    bool has(PlatformMediaSession::MediaType) const;
    int count(PlatformMediaSession::MediaType) const;
    bool activeAudioSessionRequired() const;
    bool hasActiveAudioSession() const;
    bool canProduceAudio() const;

    virtual std::optional<NowPlayingInfo> nowPlayingInfo() const;
    virtual bool hasActiveNowPlayingSession() const { return false; }
    virtual String lastUpdatedNowPlayingTitle() const { return emptyString(); }
    virtual double lastUpdatedNowPlayingDuration() const { return NAN; }
    virtual double lastUpdatedNowPlayingElapsedTime() const { return NAN; }
    virtual std::optional<MediaUniqueIdentifier> lastUpdatedNowPlayingInfoUniqueIdentifier() const { return std::nullopt; }
    virtual bool registeredAsNowPlayingApplication() const { return false; }
    virtual bool haveEverRegisteredAsNowPlayingApplication() const { return false; }
    virtual void prepareToSendUserMediaPermissionRequestForPage(Page&) { }

    bool willIgnoreSystemInterruptions() const { return m_willIgnoreSystemInterruptions; }
    void setWillIgnoreSystemInterruptions(bool ignore) { m_willIgnoreSystemInterruptions = ignore; }

    WEBCORE_EXPORT virtual void beginInterruption(PlatformMediaSession::InterruptionType);
    WEBCORE_EXPORT void endInterruption(PlatformMediaSession::EndInterruptionFlags);

    WEBCORE_EXPORT void applicationWillBecomeInactive();
    WEBCORE_EXPORT void applicationDidBecomeActive();
    WEBCORE_EXPORT void applicationWillEnterForeground(bool suspendedUnderLock);
    WEBCORE_EXPORT void applicationDidEnterBackground(bool suspendedUnderLock);
    WEBCORE_EXPORT void processWillSuspend();
    WEBCORE_EXPORT void processDidResume();

    bool mediaPlaybackIsPaused(std::optional<MediaSessionGroupIdentifier>);
    void pauseAllMediaPlaybackForGroup(std::optional<MediaSessionGroupIdentifier>);
    WEBCORE_EXPORT void stopAllMediaPlaybackForProcess();

    void suspendAllMediaPlaybackForGroup(std::optional<MediaSessionGroupIdentifier>);
    void resumeAllMediaPlaybackForGroup(std::optional<MediaSessionGroupIdentifier>);
    void suspendAllMediaBufferingForGroup(std::optional<MediaSessionGroupIdentifier>);
    void resumeAllMediaBufferingForGroup(std::optional<MediaSessionGroupIdentifier>);

    enum SessionRestrictionFlags {
        NoRestrictions = 0,
        ConcurrentPlaybackNotPermitted = 1 << 0,
        BackgroundProcessPlaybackRestricted = 1 << 1,
        BackgroundTabPlaybackRestricted = 1 << 2,
        InterruptedPlaybackNotPermitted = 1 << 3,
        InactiveProcessPlaybackRestricted = 1 << 4,
        SuspendedUnderLockPlaybackRestricted = 1 << 5,
    };
    typedef unsigned SessionRestrictions;

    WEBCORE_EXPORT void addRestriction(PlatformMediaSession::MediaType, SessionRestrictions);
    WEBCORE_EXPORT void removeRestriction(PlatformMediaSession::MediaType, SessionRestrictions);
    WEBCORE_EXPORT SessionRestrictions restrictions(PlatformMediaSession::MediaType);
    virtual void resetRestrictions();

    virtual bool sessionWillBeginPlayback(PlatformMediaSession&);

    virtual void sessionWillEndPlayback(PlatformMediaSession&, DelayCallingUpdateNowPlaying);
    virtual void sessionStateChanged(PlatformMediaSession&);
    virtual void sessionDidEndRemoteScrubbing(PlatformMediaSession&) { };
    virtual void clientCharacteristicsChanged(PlatformMediaSession&, bool) { }
    virtual void sessionCanProduceAudioChanged();

#if PLATFORM(IOS_FAMILY)
    virtual void configureWirelessTargetMonitoring() { }
#endif
    virtual bool hasWirelessTargetsAvailable() { return false; }
    virtual bool isMonitoringWirelessTargets() const { return false; }

    virtual void setCurrentSession(PlatformMediaSession&);
    PlatformMediaSession* currentSession() const;

    void sessionIsPlayingToWirelessPlaybackTargetChanged(PlatformMediaSession&);

    WEBCORE_EXPORT void setIsPlayingToAutomotiveHeadUnit(bool);
    bool isPlayingToAutomotiveHeadUnit() const { return m_isPlayingToAutomotiveHeadUnit; }

    WEBCORE_EXPORT void setSupportsSpatialAudioPlayback(bool);
    virtual std::optional<bool> supportsSpatialAudioPlaybackForConfiguration(const MediaConfiguration&);

    void forEachMatchingSession(const Function<bool(const PlatformMediaSession&)>& predicate, const Function<void(PlatformMediaSession&)>& matchingCallback);

    bool processIsSuspended() const { return m_processIsSuspended; }

    WEBCORE_EXPORT void addAudioCaptureSource(AudioCaptureSource&);
    WEBCORE_EXPORT void removeAudioCaptureSource(AudioCaptureSource&);
    void audioCaptureSourceStateChanged() { updateSessionState(); }
    bool hasAudioCaptureSource(AudioCaptureSource& source) const { return m_audioCaptureSources.contains(source); }

    WEBCORE_EXPORT void processDidReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&);

    bool isInterrupted() const { return !!m_currentInterruption; }
    bool hasNoSession() const;

    virtual void addSupportedCommand(PlatformMediaSession::RemoteControlCommandType) { };
    virtual void removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType) { };
    virtual RemoteCommandListener::RemoteCommandsSet supportedCommands() const { return { }; };

    WEBCORE_EXPORT void processSystemWillSleep();
    WEBCORE_EXPORT void processSystemDidWake();

    virtual void resetHaveEverRegisteredAsNowPlayingApplicationForTesting() { };
    virtual void resetSessionState() { };

    bool isApplicationInBackground() const { return m_isApplicationInBackground; }

    WeakPtr<PlatformMediaSession> bestEligibleSessionForRemoteControls(const Function<bool(const PlatformMediaSession&)>&, PlatformMediaSession::PlaybackControlsPurpose);

    WEBCORE_EXPORT void addNowPlayingMetadataObserver(const NowPlayingMetadataObserver&);
    WEBCORE_EXPORT void removeNowPlayingMetadataObserver(const NowPlayingMetadataObserver&);

    bool hasActiveNowPlayingSessionInGroup(std::optional<MediaSessionGroupIdentifier>);

    virtual void updatePresentingApplicationPIDIfNecessary(ProcessID) { }

protected:
    friend class PlatformMediaSession;
    static std::unique_ptr<PlatformMediaSessionManager> create();
    PlatformMediaSessionManager();

    virtual void addSession(PlatformMediaSession&);
    virtual void removeSession(PlatformMediaSession&);

    void forEachSession(const Function<void(PlatformMediaSession&)>&);
    void forEachSessionInGroup(std::optional<MediaSessionGroupIdentifier>, const Function<void(PlatformMediaSession&)>&);
    bool anyOfSessions(const Function<bool(const PlatformMediaSession&)>&) const;

    void maybeDeactivateAudioSession();
    bool maybeActivateAudioSession();

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return 0; }
    ASCIILiteral logClassName() const override { return "PlatformMediaSessionManager"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    int countActiveAudioCaptureSources();

    bool computeSupportsSeeking() const;

    std::optional<bool> supportsSpatialAudioPlayback() { return m_supportsSpatialAudioPlayback; }

    void nowPlayingMetadataChanged(const NowPlayingMetadata&);
    void enqueueTaskOnMainThread(Function<void()>&&);

private:
    friend class Internals;

    void scheduleUpdateSessionState();
    virtual void updateSessionState() { }

    Vector<WeakPtr<PlatformMediaSession>> sessionsMatching(const Function<bool(const PlatformMediaSession&)>&) const;
    WeakPtr<PlatformMediaSession> firstSessionMatching(const Function<bool(const PlatformMediaSession&)>&) const;

#if !RELEASE_LOG_DISABLED
    void scheduleStateLog();
    void dumpSessionStates();
#endif

    std::array<SessionRestrictions, static_cast<unsigned>(PlatformMediaSession::MediaType::WebAudio) + 1> m_restrictions;
    mutable Vector<WeakPtr<PlatformMediaSession>> m_sessions;

    std::optional<PlatformMediaSession::InterruptionType> m_currentInterruption;
    mutable bool m_isApplicationInBackground { false };
    bool m_willIgnoreSystemInterruptions { false };
    bool m_processIsSuspended { false };
    bool m_isPlayingToAutomotiveHeadUnit { false };
    std::optional<bool> m_supportsSpatialAudioPlayback;

    bool m_alreadyScheduledSessionStatedUpdate { false };
#if USE(AUDIO_SESSION)
    bool m_becameActive { false };
#endif

    WeakHashSet<AudioCaptureSource> m_audioCaptureSources;
    bool m_hasScheduledSessionStateUpdate { false };

    WeakHashSet<NowPlayingMetadataObserver> m_nowPlayingMetadataObservers;
    TaskCancellationGroup m_taskGroup;

#if ENABLE(ALTERNATE_WEBM_PLAYER)
    static bool m_alternateWebMPlayerEnabled;
#endif
#if HAVE(SC_CONTENT_SHARING_PICKER)
    static bool s_useSCContentSharingPicker;
#endif

#if ENABLE(VP9)
    static bool m_vp9DecoderEnabled;
    static bool m_swVPDecodersAlwaysEnabled;
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
    static bool s_mediaCapabilityGrantsEnabled;
#endif

#if !RELEASE_LOG_DISABLED
    UniqueRef<Timer> m_stateLogTimer;
    Ref<AggregateLogger> m_logger;
#endif
};

} // namespace WebCore
