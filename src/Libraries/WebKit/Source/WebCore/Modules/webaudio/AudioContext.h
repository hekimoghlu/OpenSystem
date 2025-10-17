/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include "BaseAudioContext.h"
#include "DefaultAudioDestinationNode.h"
#include "MediaCanStartListener.h"
#include "MediaProducer.h"
#include "MediaUniqueIdentifier.h"
#include "PlatformMediaSession.h"
#include <wtf/UniqueRef.h>

namespace WebCore {

class LocalDOMWindow;
class HTMLMediaElement;
class MediaStream;
class MediaStreamAudioDestinationNode;
class MediaStreamAudioSourceNode;

struct AudioContextOptions;
struct AudioTimestamp;

class AudioContext final
    : public BaseAudioContext
    , public MediaProducer
    , public MediaCanStartListener
    , private PlatformMediaSessionClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AudioContext);
public:
    // Create an AudioContext for rendering to the audio hardware.
    static ExceptionOr<Ref<AudioContext>> create(Document&, AudioContextOptions&&);
    ~AudioContext();

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    WEBCORE_EXPORT static void setDefaultSampleRateForTesting(std::optional<float>);

    void close(DOMPromiseDeferred<void>&&);

    DefaultAudioDestinationNode& destination() final { return m_destinationNode.get(); }
    const DefaultAudioDestinationNode& destination() const final { return m_destinationNode.get(); }

    double baseLatency();
    double outputLatency();

    AudioTimestamp getOutputTimestamp();

#if ENABLE(VIDEO)
    ExceptionOr<Ref<MediaElementAudioSourceNode>> createMediaElementSource(HTMLMediaElement&);
#endif
#if ENABLE(MEDIA_STREAM)
    ExceptionOr<Ref<MediaStreamAudioSourceNode>> createMediaStreamSource(MediaStream&);
    ExceptionOr<Ref<MediaStreamAudioDestinationNode>> createMediaStreamDestination();
#endif

    void suspendRendering(DOMPromiseDeferred<void>&&);
    void resumeRendering(DOMPromiseDeferred<void>&&);

    void sourceNodeWillBeginPlayback(AudioNode&) final;
    void lazyInitialize() final;

    void startRendering();

    void isPlayingAudioDidChange();

    // Restrictions to change default behaviors.
    enum BehaviorRestrictionFlags {
        NoRestrictions = 0,
        RequireUserGestureForAudioStartRestriction = 1 << 0,
        RequirePageConsentForAudioStartRestriction = 1 << 1,
    };
    typedef unsigned BehaviorRestrictions;
    BehaviorRestrictions behaviorRestrictions() const { return m_restrictions; }
    void addBehaviorRestriction(BehaviorRestrictions restriction) { m_restrictions |= restriction; }
    void removeBehaviorRestriction(BehaviorRestrictions restriction) { m_restrictions &= ~restriction; }

    void defaultDestinationWillBecomeConnected();

private:
    AudioContext(Document&, const AudioContextOptions&);

    bool willBeginPlayback();

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final;
    uint64_t logIdentifier() const final { return BaseAudioContext::logIdentifier(); }
#endif

    void constructCommon();

    bool userGestureRequiredForAudioStart() const { return m_restrictions & RequireUserGestureForAudioStartRestriction; }
    bool pageConsentRequiredForAudioStart() const { return m_restrictions & RequirePageConsentForAudioStartRestriction; }

    bool willPausePlayback();

    void uninitialize() final;
    bool isOfflineContext() const final { return false; }

    // MediaProducer
    MediaProducerMediaStateFlags mediaState() const final;
    void pageMutedStateDidChange() final;

    // PlatformMediaSessionClient
    PlatformMediaSession::MediaType mediaType() const final { return isSuspended() || isStopped() ? PlatformMediaSession::MediaType::None : PlatformMediaSession::MediaType::WebAudio; }
    PlatformMediaSession::MediaType presentationType() const final { return PlatformMediaSession::MediaType::WebAudio; }
    void mayResumePlayback(bool shouldResume) final;
    void suspendPlayback() final;
    bool canReceiveRemoteControlCommands() const final;
    void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&) final;
    bool supportsSeeking() const final { return false; }
    bool shouldOverrideBackgroundPlaybackRestriction(PlatformMediaSession::InterruptionType) const final;
    bool canProduceAudio() const final { return true; }
    bool isSuspended() const final;
    bool isPlaying() const final;
    bool isAudible() const final;
    std::optional<MediaSessionGroupIdentifier> mediaSessionGroupIdentifier() const final;
    bool isNowPlayingEligible() const final;
    std::optional<NowPlayingInfo> nowPlayingInfo() const final;
    WeakPtr<PlatformMediaSession> selectBestMediaSession(const Vector<WeakPtr<PlatformMediaSession>>&, PlatformMediaSession::PlaybackControlsPurpose) final;
    void isActiveNowPlayingSessionChanged() final;
    ProcessID presentingApplicationPID() const final;

    // MediaCanStartListener.
    void mediaCanStart(Document&) final;

    // ActiveDOMObject
    void suspend(ReasonForSuspension) final;
    void resume() final;
    bool virtualHasPendingActivity() const final;

    UniqueRef<DefaultAudioDestinationNode> m_destinationNode;
    std::unique_ptr<PlatformMediaSession> m_mediaSession;
    MediaUniqueIdentifier m_currentIdentifier;

    BehaviorRestrictions m_restrictions { NoRestrictions };

    // [[suspended by user]] flag in the specification:
    // https://www.w3.org/TR/webaudio/#dom-audiocontext-suspended-by-user-slot
    bool m_wasSuspendedByScript { false };

    bool m_canOverrideBackgroundPlaybackRestriction { true };
};

} // WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioContext)
    static bool isType(const WebCore::BaseAudioContext& context) { return !context.isOfflineContext(); }
SPECIALIZE_TYPE_TRAITS_END()
