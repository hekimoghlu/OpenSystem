/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
#include "MediaRecorder.h"

#if ENABLE(MEDIA_RECORDER)

#include "Blob.h"
#include "BlobEvent.h"
#include "Document.h"
#include "EventNames.h"
#include "MediaRecorderErrorEvent.h"
#include "MediaRecorderPrivate.h"
#include "Page.h"
#include "SharedBuffer.h"
#include "WindowEventLoop.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA) && USE(AVFOUNDATION)
#include "MediaRecorderPrivateAVFImpl.h"
#endif

#if USE(GSTREAMER_TRANSCODER)
#include "MediaRecorderPrivateGStreamer.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaRecorder);

MediaRecorder::CreatorFunction MediaRecorder::m_customCreator = nullptr;

bool MediaRecorder::isTypeSupported(Document& document, const String& value)
{
#if PLATFORM(COCOA) || USE(GSTREAMER_TRANSCODER)
    if (value.isEmpty())
        return true;

    ContentType mimeType(value);
#if PLATFORM(COCOA)
    return MediaRecorderPrivateAVFImpl::isTypeSupported(document, mimeType);
#elif USE(GSTREAMER_TRANSCODER)
    UNUSED_PARAM(document);
    return MediaRecorderPrivateGStreamer::isTypeSupported(mimeType);
#endif
#else
    UNUSED_PARAM(document);
    UNUSED_PARAM(value);
    return false;
#endif
}

ExceptionOr<Ref<MediaRecorder>> MediaRecorder::create(Document& document, Ref<MediaStream>&& stream, Options&& options)
{
    auto* page = document.page();
    if (!page)
        return Exception { ExceptionCode::InvalidStateError };

    if (!isTypeSupported(document, options.mimeType))
        return Exception { ExceptionCode::NotSupportedError, "mimeType is not supported"_s };

    auto recorder = adoptRef(*new MediaRecorder(document, WTFMove(stream), WTFMove(options)));
    recorder->suspendIfNeeded();
    return recorder;
}

void MediaRecorder::setCustomPrivateRecorderCreator(CreatorFunction creator)
{
    m_customCreator = creator;
}

ExceptionOr<std::unique_ptr<MediaRecorderPrivate>> MediaRecorder::createMediaRecorderPrivate(MediaStreamPrivate& stream, const Options& options)
{
    if (m_customCreator)
        return m_customCreator(stream, options);

#if PLATFORM(COCOA) && USE(AVFOUNDATION)
    std::unique_ptr<MediaRecorderPrivate> result = MediaRecorderPrivateAVFImpl::create(stream, options);
#elif USE(GSTREAMER_TRANSCODER)
    std::unique_ptr<MediaRecorderPrivate> result = MediaRecorderPrivateGStreamer::create(stream, options);
#else
    std::unique_ptr<MediaRecorderPrivate> result;
#endif
    if (!result)
        return Exception { ExceptionCode::NotSupportedError, "The MediaRecorder is unsupported on this platform"_s };
    return result;
}

MediaRecorder::MediaRecorder(Document& document, Ref<MediaStream>&& stream, Options&& options)
    : ActiveDOMObject(document)
    , m_options(WTFMove(options))
    , m_stream(WTFMove(stream))
    , m_timeSliceTimer([this] { Ref { *this }->requestDataInternal(ReturnDataIfEmpty::No); })
{
    computeInitialBitRates();

    m_tracks = m_stream->privateStream().tracks();
    m_stream->privateStream().addObserver(*this);
}

MediaRecorder::~MediaRecorder()
{
    m_stream->privateStream().removeObserver(*this);
    stopRecordingInternal();
}

Document* MediaRecorder::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

void MediaRecorder::stop()
{
    m_isActive = false;
    stopRecordingInternal();
}

void MediaRecorder::suspend(ReasonForSuspension reason)
{
    if (reason != ReasonForSuspension::BackForwardCache)
        return;

    if (!m_isActive || state() == RecordingState::Inactive)
        return;

    stopRecordingInternal();

    queueTaskToDispatchEvent(*this, TaskSource::Networking, MediaRecorderErrorEvent::create(eventNames().errorEvent, Exception { ExceptionCode::UnknownError, "MediaStream recording was interrupted"_s }));
}

ExceptionOr<void> MediaRecorder::startRecording(std::optional<unsigned> timeSlice)
{
    if (!m_isActive)
        return Exception { ExceptionCode::InvalidStateError, "The MediaRecorder is not active"_s };

    if (state() != RecordingState::Inactive)
        return Exception { ExceptionCode::InvalidStateError, "The MediaRecorder's state must be inactive in order to start recording"_s };

    updateBitRates();

    Options options { m_options };
    options.audioBitsPerSecond = m_audioBitsPerSecond;
    options.videoBitsPerSecond = m_videoBitsPerSecond;

    ASSERT(!m_private);
    auto result = createMediaRecorderPrivate(m_stream->privateStream(), options);

    if (result.hasException())
        return result.releaseException();

    m_private = result.releaseReturnValue();
    m_private->startRecording([this, weakThis = WeakPtr { *this }, pendingActivity = makePendingActivity(*this)](auto&& mimeTypeOrException, unsigned audioBitsPerSecond, unsigned videoBitsPerSecond) mutable {
        auto protectedThis = RefPtr { weakThis.get() };
        if (!protectedThis)
            return;

        if (!m_isActive)
            return;

        if (mimeTypeOrException.hasException()) {
            stopRecordingInternal();
            queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, exception = mimeTypeOrException.releaseException()]() mutable {
                if (!m_isActive)
                    return;
                dispatchError(WTFMove(exception));
            });
            return;
        }

        queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this, mimeType = mimeTypeOrException.releaseReturnValue(), audioBitsPerSecond, videoBitsPerSecond]() mutable {
            if (!m_isActive)
                return;
            m_options.mimeType = WTFMove(mimeType);
            m_options.audioBitsPerSecond = audioBitsPerSecond;
            m_options.videoBitsPerSecond = videoBitsPerSecond;

            dispatchEvent(Event::create(eventNames().startEvent, Event::CanBubble::No, Event::IsCancelable::No));
        });
    });

    for (auto& track : m_tracks)
        track->addObserver(*this);

    m_state = RecordingState::Recording;
    m_timeSlice = timeSlice ? std::make_optional(std::max(m_mimimumTimeSlice, *timeSlice)) : std::nullopt;
    if (m_timeSlice)
        m_timeSliceTimer.startOneShot(Seconds::fromMilliseconds(*m_timeSlice));
    return { };
}

static inline Ref<BlobEvent> createDataAvailableEvent(ScriptExecutionContext* context, RefPtr<FragmentedSharedBuffer>&& buffer, const String& mimeType, double timeCode)
{
    auto blob = buffer ? Blob::create(context, buffer->extractData(), mimeType) : Blob::create(context);
    return BlobEvent::create(eventNames().dataavailableEvent, BlobEvent::Init { { false, false, false }, WTFMove(blob), timeCode }, BlobEvent::IsTrusted::Yes);
}

void MediaRecorder::stopRecording()
{
    if (state() == RecordingState::Inactive)
        return;

    updateBitRates();

    stopRecordingInternal();
    fetchData([this](RefPtr<FragmentedSharedBuffer>&& buffer, auto& mimeType, auto timeCode) {
        if (!m_isActive)
            return;

        dispatchEvent(createDataAvailableEvent(scriptExecutionContext(), WTFMove(buffer), mimeType, timeCode));

        if (!m_isActive)
            return;
        dispatchEvent(Event::create(eventNames().stopEvent, Event::CanBubble::No, Event::IsCancelable::No));
    }, TakePrivateRecorder::Yes);
    return;
}

ExceptionOr<void> MediaRecorder::requestData()
{
    return requestDataInternal(ReturnDataIfEmpty::Yes);
}

ExceptionOr<void> MediaRecorder::requestDataInternal(ReturnDataIfEmpty returnDataIfEmpty)
{
    if (state() == RecordingState::Inactive)
        return Exception { ExceptionCode::InvalidStateError, "The MediaRecorder's state cannot be inactive"_s };

    if (m_timeSliceTimer.isActive())
        m_timeSliceTimer.stop();

    fetchData([this, returnDataIfEmpty](auto&& buffer, auto& mimeType, auto timeCode) {
        if (!m_isActive)
            return;

        if (returnDataIfEmpty == ReturnDataIfEmpty::Yes || !buffer->isEmpty())
            dispatchEvent(createDataAvailableEvent(scriptExecutionContext(), WTFMove(buffer), mimeType, timeCode));

        switch (state()) {
        case RecordingState::Inactive:
            break;
        case RecordingState::Recording:
            ASSERT(m_isActive);
            if (m_timeSlice)
                m_timeSliceTimer.startOneShot(Seconds::fromMilliseconds(*m_timeSlice));
            break;
        case RecordingState::Paused:
            if (m_timeSlice)
                m_nextFireInterval = Seconds::fromMilliseconds(*m_timeSlice);
            break;
        }
    }, TakePrivateRecorder::No);
    return { };
}

ExceptionOr<void> MediaRecorder::pauseRecording()
{
    if (state() == RecordingState::Inactive)
        return Exception { ExceptionCode::InvalidStateError, "The MediaRecorder's state cannot be inactive"_s };

    if (state() == RecordingState::Paused)
        return { };

    m_state = RecordingState::Paused;

    if (m_timeSliceTimer.isActive()) {
        m_nextFireInterval = m_timeSliceTimer.nextFireInterval();
        m_timeSliceTimer.stop();
    }

    m_private->pause([this, pendingActivity = makePendingActivity(*this)]() {
        if (!m_isActive)
            return;
        queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this]() mutable {
            if (!m_isActive)
                return;
            dispatchEvent(Event::create(eventNames().pauseEvent, Event::CanBubble::No, Event::IsCancelable::No));
        });
    });
    return { };
}

ExceptionOr<void> MediaRecorder::resumeRecording()
{
    if (state() == RecordingState::Inactive)
        return Exception { ExceptionCode::InvalidStateError, "The MediaRecorder's state cannot be inactive"_s };

    if (state() == RecordingState::Recording)
        return { };

    m_state = RecordingState::Recording;

    if (m_nextFireInterval) {
        m_timeSliceTimer.startOneShot(*m_nextFireInterval);
        m_nextFireInterval = { };
    }

    m_private->resume([this, pendingActivity = makePendingActivity(*this)]() {
        if (!m_isActive)
            return;
        queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this]() mutable {
            if (!m_isActive)
                return;
            dispatchEvent(Event::create(eventNames().resumeEvent, Event::CanBubble::No, Event::IsCancelable::No));
        });
    });
    return { };
}

void MediaRecorder::fetchData(FetchDataCallback&& callback, TakePrivateRecorder takeRecorder)
{
    auto& privateRecorder = *m_private;

    std::unique_ptr<MediaRecorderPrivate> takenPrivateRecorder;
    if (takeRecorder == TakePrivateRecorder::Yes)
        takenPrivateRecorder = WTFMove(m_private);

    auto fetchDataCallback = [this, privateRecorder = WTFMove(takenPrivateRecorder), callback = WTFMove(callback)](RefPtr<FragmentedSharedBuffer>&& buffer, auto& mimeType, auto timeCode) mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [buffer = WTFMove(buffer), mimeType, timeCode, callback = WTFMove(callback)]() mutable {
            callback(WTFMove(buffer), mimeType, timeCode);
        });
    };

    if (m_isFetchingData) {
        m_pendingFetchDataTasks.append(WTFMove(fetchDataCallback));
        return;
    }

    m_isFetchingData = true;
    privateRecorder.fetchData([this, pendingActivity = makePendingActivity(*this), callback = WTFMove(fetchDataCallback)](RefPtr<FragmentedSharedBuffer>&& buffer, auto& mimeType, auto timeCode) mutable {
        m_isFetchingData = false;
        callback(WTFMove(buffer), mimeType, timeCode);
        for (auto& task : std::exchange(m_pendingFetchDataTasks, { }))
            task({ }, mimeType, timeCode);
    });
}

void MediaRecorder::stopRecordingInternal(CompletionHandler<void()>&& completionHandler)
{
    if (state() == RecordingState::Inactive) {
        completionHandler();
        return;
    }

    for (auto& track : m_tracks)
        track->removeObserver(*this);

    m_state = RecordingState::Inactive;
    m_private->stop(WTFMove(completionHandler));
}

void MediaRecorder::handleTrackChange()
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this] {
        stopRecordingInternal([this, pendingActivity = makePendingActivity(*this)] {
            queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this] {
                if (!m_isActive)
                    return;
                dispatchError(Exception { ExceptionCode::InvalidModificationError, "Track cannot be added to or removed from the MediaStream while recording"_s });

                if (!m_isActive)
                    return;
                dispatchEvent(createDataAvailableEvent(scriptExecutionContext(), { }, { }, 0));

                if (!m_isActive)
                    return;
                dispatchEvent(Event::create(eventNames().stopEvent, Event::CanBubble::No, Event::IsCancelable::No));
            });
        });
    });
}

void MediaRecorder::dispatchError(Exception&& exception)
{
    if (!m_isActive)
        return;
    dispatchEvent(MediaRecorderErrorEvent::create(eventNames().errorEvent, WTFMove(exception)));
}

void MediaRecorder::trackEnded(MediaStreamTrackPrivate&)
{
    auto position = m_tracks.findIf([](auto& track) {
        return !track->ended();
    });
    if (position != notFound)
        return;

    queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this] {
        stopRecordingInternal([this, pendingActivity = makePendingActivity(*this)] {
            queueTaskKeepingObjectAlive(*this, TaskSource::Networking, [this] {
                if (!m_isActive)
                    return;
                dispatchEvent(createDataAvailableEvent(scriptExecutionContext(), { }, { }, 0));

                if (!m_isActive)
                    return;
                dispatchEvent(Event::create(eventNames().stopEvent, Event::CanBubble::No, Event::IsCancelable::No));
            });
        });
    });
}

void MediaRecorder::trackMutedChanged(MediaStreamTrackPrivate& track)
{
    if (m_private)
        m_private->trackMutedChanged(track);
}

void MediaRecorder::trackEnabledChanged(MediaStreamTrackPrivate& track)
{
    if (m_private)
        m_private->trackEnabledChanged(track);
}

bool MediaRecorder::virtualHasPendingActivity() const
{
    return m_state != RecordingState::Inactive;
}

void MediaRecorder::computeBitRates(const MediaStreamPrivate* stream)
{
    auto bitRates = MediaRecorderPrivate::computeBitRates(m_options, stream);
    m_audioBitsPerSecond = bitRates.audio;
    m_videoBitsPerSecond = bitRates.video;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
