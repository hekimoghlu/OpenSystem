/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#if ENABLE(MEDIA_RECORDER)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "MediaRecorderPrivateOptions.h"
#include "MediaStream.h"
#include "MediaStreamTrackPrivate.h"
#include "Timer.h"
#include <wtf/Deque.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class Blob;
class Document;
class MediaRecorderPrivate;

class MediaRecorder final
    : public ActiveDOMObject
    , public RefCounted<MediaRecorder>
    , public EventTarget
    , private MediaStreamPrivateObserver
    , private MediaStreamTrackPrivateObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaRecorder);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    enum class RecordingState { Inactive, Recording, Paused };
    
    ~MediaRecorder();
    
    static bool isTypeSupported(Document&, const String&);

    using Options = MediaRecorderPrivateOptions;
    static ExceptionOr<Ref<MediaRecorder>> create(Document&, Ref<MediaStream>&&, Options&& = { });

    using CreatorFunction = ExceptionOr<std::unique_ptr<MediaRecorderPrivate>> (*)(MediaStreamPrivate&, const Options&);

    WEBCORE_EXPORT static void setCustomPrivateRecorderCreator(CreatorFunction);

    RecordingState state() const { return m_state; }
    const String& mimeType() const { return m_options.mimeType; }

    ExceptionOr<void> startRecording(std::optional<unsigned>);
    void stopRecording();
    ExceptionOr<void> requestData();
    ExceptionOr<void> pauseRecording();
    ExceptionOr<void> resumeRecording();

    unsigned videoBitsPerSecond() const { return m_videoBitsPerSecond; }
    unsigned audioBitsPerSecond() const { return m_audioBitsPerSecond; }

    MediaStream& stream() { return m_stream.get(); }

    USING_CAN_MAKE_WEAKPTR(EventTarget);

private:
    MediaRecorder(Document&, Ref<MediaStream>&&, Options&&);

    static ExceptionOr<std::unique_ptr<MediaRecorderPrivate>> createMediaRecorderPrivate(MediaStreamPrivate&, const Options&);
    
    Document* document() const;

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaRecorder; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    // ActiveDOMObject.
    void suspend(ReasonForSuspension) final;
    void stop() final;
    bool virtualHasPendingActivity() const final;
    
    void stopRecordingInternal(CompletionHandler<void()>&& = [] { });
    void dispatchError(Exception&&);

    enum class TakePrivateRecorder : bool { No, Yes };
    using FetchDataCallback = Function<void(RefPtr<FragmentedSharedBuffer>&&, const String& mimeType, double)>;
    void fetchData(FetchDataCallback&&, TakePrivateRecorder);
    enum class ReturnDataIfEmpty : bool { No, Yes };
    ExceptionOr<void> requestDataInternal(ReturnDataIfEmpty);

    // MediaStream::Observer
    void didAddTrack(MediaStreamTrackPrivate&) final { handleTrackChange(); }
    void didRemoveTrack(MediaStreamTrackPrivate&) final { handleTrackChange(); }

    void handleTrackChange();

    // MediaStreamTrackPrivateObserver
    void trackEnded(MediaStreamTrackPrivate&) final;
    void trackMutedChanged(MediaStreamTrackPrivate&) final;
    void trackEnabledChanged(MediaStreamTrackPrivate&) final;
    void trackSettingsChanged(MediaStreamTrackPrivate&) final { };

    void computeInitialBitRates() { computeBitRates(nullptr); }
    void updateBitRates() { computeBitRates(&m_stream->privateStream()); }
    void computeBitRates(const MediaStreamPrivate*);

    static CreatorFunction m_customCreator;

    Options m_options;
    Ref<MediaStream> m_stream;
    std::unique_ptr<MediaRecorderPrivate> m_private;
    RecordingState m_state { RecordingState::Inactive };
    Vector<Ref<MediaStreamTrackPrivate>> m_tracks;
    static constexpr unsigned m_mimimumTimeSlice { 100 };
    std::optional<unsigned> m_timeSlice;
    Timer m_timeSliceTimer;

    bool m_isActive { true };
    bool m_isFetchingData { false };
    Deque<FetchDataCallback> m_pendingFetchDataTasks;

    unsigned m_audioBitsPerSecond { 0 };
    unsigned m_videoBitsPerSecond { 0 };

    std::optional<Seconds> m_nextFireInterval;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
