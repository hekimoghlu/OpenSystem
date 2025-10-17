/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "ActiveDOMObject.h"
#include "Blob.h"
#include "EventTarget.h"
#include "IDLTypes.h"
#include "JSDOMPromiseDeferred.h"
#include "MediaProducer.h"
#include "MediaStreamTrackDataHolder.h"
#include "MediaStreamTrackPrivate.h"
#include "MediaTrackCapabilities.h"
#include "MediaTrackConstraints.h"
#include "PhotoCapabilities.h"
#include "PhotoSettings.h"
#include "PlatformMediaSession.h"
#include <wtf/LoggerHelper.h>

namespace WebCore {

class AudioSourceProvider;
class Document;

struct MediaTrackConstraints;

class MediaStreamTrack
    : public RefCounted<MediaStreamTrack>
    , public ActiveDOMObject
    , public EventTarget
    , private MediaStreamTrackPrivateObserver
    , private AudioCaptureSource
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaStreamTrack);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    class Observer {
    public:
        virtual ~Observer() = default;
        virtual void trackDidEnd() = 0;
    };

    enum class RegisterCaptureTrackToOwner : bool { No, Yes };
    static Ref<MediaStreamTrack> create(ScriptExecutionContext&, Ref<MediaStreamTrackPrivate>&&, RegisterCaptureTrackToOwner = RegisterCaptureTrackToOwner::Yes);
    static Ref<MediaStreamTrack> create(ScriptExecutionContext&, UniqueRef<MediaStreamTrackDataHolder>&&);
    virtual ~MediaStreamTrack();

    static MediaProducerMediaStateFlags captureState(const RealtimeMediaSource&);

    virtual bool isCanvas() const { return false; }

    const AtomString& kind() const;
    WEBCORE_EXPORT const String& id() const;
    const String& label() const;

    const AtomString& contentHint() const;
    void setContentHint(const String&);
        
    bool enabled() const;
    void setEnabled(bool);

    bool muted() const;
    bool mutedForBindings() const;

    enum class State { Live, Ended };
    State readyState() const { return m_readyState; }

    bool ended() const;

    virtual RefPtr<MediaStreamTrack> clone();

    enum class StopMode { Silently, PostEvent };
    void stopTrack(StopMode = StopMode::Silently);

    bool isCaptureTrack() const { return m_isCaptureTrack; }
    bool isVideo() const { return m_private->isVideo(); }
    bool isAudio() const { return m_private->isAudio(); }

    struct TrackSettings {
        std::optional<int> width;
        std::optional<int> height;
        std::optional<double> aspectRatio;
        std::optional<double> frameRate;
        String facingMode;
        std::optional<double> volume;
        std::optional<int> sampleRate;
        std::optional<int> sampleSize;
        std::optional<bool> echoCancellation;
        String displaySurface;
        String deviceId;
        String groupId;

        String whiteBalanceMode;
        std::optional<double> zoom;
        std::optional<bool> torch;
        std::optional<bool> backgroundBlur;
        std::optional<bool> powerEfficient;
    };
    TrackSettings getSettings() const;

    using TrackCapabilities = MediaTrackCapabilities;
    TrackCapabilities getCapabilities() const;

    using TakePhotoPromise = NativePromise<std::pair<Vector<uint8_t>, String>, Exception>;
    Ref<TakePhotoPromise> takePhoto(PhotoSettings&&);

    using PhotoCapabilitiesPromise = NativePromise<PhotoCapabilities, Exception>;
    Ref<PhotoCapabilitiesPromise> getPhotoCapabilities();

    using PhotoSettingsPromise = NativePromise<PhotoSettings, Exception>;
    Ref<PhotoSettingsPromise> getPhotoSettings();

    const MediaTrackConstraints& getConstraints() const { return m_constraints; }
    void setConstraints(MediaTrackConstraints&& constraints) { m_constraints = WTFMove(constraints); }

    void applyConstraints(const std::optional<MediaTrackConstraints>&, DOMPromiseDeferred<void>&&);

    RealtimeMediaSource& source() const { return m_private->source(); }
    RealtimeMediaSource& sourceForProcessor() const { return m_private->sourceForProcessor(); }
    MediaStreamTrackPrivate& privateTrack() { return m_private.get(); }
    const MediaStreamTrackPrivate& privateTrack() const { return m_private.get(); }

#if ENABLE(WEB_AUDIO)
    RefPtr<WebAudioSourceProvider> createAudioSourceProvider();
#endif

    MediaProducerMediaStateFlags mediaState() const;

    void addObserver(Observer&);
    void removeObserver(Observer&);

    void setIdForTesting(String&& id) { m_private->setIdForTesting(WTFMove(id)); }

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_private->logger(); }
    uint64_t logIdentifier() const final { return m_private->logIdentifier(); }
#endif

    void setShouldFireMuteEventImmediately(bool value) { m_shouldFireMuteEventImmediately = value; }

    struct Storage {
        bool enabled { false };
        bool ended { false };
        bool muted { false };
        RealtimeMediaSourceSettings settings;
        RealtimeMediaSourceCapabilities capabilities;
        RefPtr<RealtimeMediaSource> source;
    };

    bool isDetached() const { return m_isDetached; }
    UniqueRef<MediaStreamTrackDataHolder> detach();

    void setMediaStreamId(const String& id) { m_mediaStreamId = id; }
    const String& mediaStreamId() const { return m_mediaStreamId; }

protected:
    MediaStreamTrack(ScriptExecutionContext&, Ref<MediaStreamTrackPrivate>&&);

    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
        
    Ref<MediaStreamTrackPrivate> m_private;
        
private:
    explicit MediaStreamTrack(MediaStreamTrack&);

    void configureTrackRendering();

    // ActiveDOMObject.
    void stop() final { stopTrack(); }
    void suspend(ReasonForSuspension) final;
    bool virtualHasPendingActivity() const final;

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaStreamTrack; }

    // MediaStreamTrackPrivateObserver
    void trackStarted(MediaStreamTrackPrivate&) final;
    void trackEnded(MediaStreamTrackPrivate&) final;
    void trackMutedChanged(MediaStreamTrackPrivate&) final;
    void trackSettingsChanged(MediaStreamTrackPrivate&) final;
    void trackEnabledChanged(MediaStreamTrackPrivate&) final;
    void trackConfigurationChanged(MediaStreamTrackPrivate&) final;

    // AudioCaptureSource
    bool isCapturingAudio() const final;
    bool wantsToCaptureAudio() const final;

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "MediaStreamTrack"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    Vector<Observer*> m_observers;

    MediaTrackConstraints m_constraints;

    String m_mediaStreamId;
    State m_readyState { State::Live };
    bool m_muted { false };
    bool m_ended { false };
    const bool m_isCaptureTrack { false };
    bool m_isInterrupted { false };
    bool m_shouldFireMuteEventImmediately { false };
    bool m_isDetached { false };
    mutable AtomString m_kind;
    mutable AtomString m_contentHint;
};

typedef Vector<Ref<MediaStreamTrack>> MediaStreamTrackVector;

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
