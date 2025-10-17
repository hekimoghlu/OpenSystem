/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#if ENABLE(MEDIA_STREAM) && ENABLE(WEB_CODECS)

#include "ExceptionOr.h"
#include "MediaStreamTrack.h"
#include "ReadableStreamSource.h"
#include "RealtimeMediaSource.h"
#include "WebCodecsVideoFrame.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {
class JSGlobaObject;
}

namespace WebCore {

class ReadableStream;
class ScriptExecutionContext;
class WebCodecsVideoFrame;

class MediaStreamTrackProcessor
    : public RefCounted<MediaStreamTrackProcessor>
    , public CanMakeWeakPtr<MediaStreamTrackProcessor>
    , private ContextDestructionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaStreamTrackProcessor);
public:
    struct Init {
        RefPtr<MediaStreamTrack> track;
        unsigned short maxBufferSize { 1 };
    };

    static ExceptionOr<Ref<MediaStreamTrackProcessor>> create(ScriptExecutionContext&, Init&&);
    ~MediaStreamTrackProcessor();

    ExceptionOr<Ref<ReadableStream>> readable(JSC::JSGlobalObject&);

    // Lives in ScriptExecutionContext only.
    class Source final
        : public ReadableStreamSource
        , public MediaStreamTrackPrivateObserver {
        WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Source);
    public:
        Source(Ref<MediaStreamTrackPrivate>&&, MediaStreamTrackProcessor&);
        ~Source();

        bool isWaiting() const { return m_isWaiting; }
        bool isCancelled() const { return m_isCancelled; }
        void close();
        void enqueue(WebCodecsVideoFrame&, ScriptExecutionContext&);

        void ref() const final { m_processor->ref(); };
        void deref() const final { m_processor->deref(); };

        void setAsCancelled() { m_isCancelled = true; }

    private:

        // MediaStreamTrackPrivateObserver
        void trackEnded(MediaStreamTrackPrivate&) final;
        void trackMutedChanged(MediaStreamTrackPrivate&) final { }
        void trackSettingsChanged(MediaStreamTrackPrivate&) final { }
        void trackEnabledChanged(MediaStreamTrackPrivate&) final { }

        // ReadableStreamSource
        void setActive() { };
        void setInactive() { };
        void doStart() final;
        void doPull() final;
        void doCancel() final;

        bool m_isWaiting { false };
        bool m_isCancelled { false };
        const Ref<MediaStreamTrackPrivate> m_privateTrack;
        const WeakRef<MediaStreamTrackProcessor> m_processor;
    };
    using MediaStreamTrackProcessorSource = MediaStreamTrackProcessor::Source;


private:
    MediaStreamTrackProcessor(ScriptExecutionContext&, Ref<MediaStreamTrack>&&, unsigned short maxVideoFramesCount);

    // ContextDestructionObserver
    void contextDestroyed() final;

    void stopVideoFrameObserver();
    void tryEnqueueingVideoFrame();

    class VideoFrameObserver final : private RealtimeMediaSource::VideoFrameObserver {
        WTF_MAKE_TZONE_ALLOCATED(VideoFrameObserver);
    public:
        explicit VideoFrameObserver(ScriptExecutionContextIdentifier, WeakPtr<MediaStreamTrackProcessor>&&, Ref<RealtimeMediaSource>&&, unsigned short maxVideoFramesCount);
        ~VideoFrameObserver();

        void start();
        RefPtr<WebCodecsVideoFrame> takeVideoFrame(ScriptExecutionContext&);

        bool isContextThread() const { return ScriptExecutionContext::isContextThread(m_contextIdentifier); }

    private:
        // RealtimeMediaSource::VideoFrameObserver
        void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) final;

        bool m_isStarted WTF_GUARDED_BY_CAPABILITY(mainThread) { false };
        RefPtr<RealtimeMediaSource> m_realtimeVideoSource WTF_GUARDED_BY_CAPABILITY(mainThread);

        // Accessed on either thread
        const ScriptExecutionContextIdentifier m_contextIdentifier;
        const WeakPtr<MediaStreamTrackProcessor> m_processor;
        Lock m_videoFramesLock;
        Deque<Ref<VideoFrame>> m_videoFrames WTF_GUARDED_BY_LOCK(m_videoFramesLock);
        const unsigned short m_maxVideoFramesCount { 1 };
    };

    class VideoFrameObserverWrapper : public ThreadSafeRefCounted<VideoFrameObserverWrapper, WTF::DestructionThread::Main> {
    public:
        static Ref<VideoFrameObserverWrapper> create(ScriptExecutionContextIdentifier, MediaStreamTrackProcessor&, Ref<RealtimeMediaSource>&&, unsigned short maxVideoFramesCount);

        void start();
        RefPtr<WebCodecsVideoFrame> takeVideoFrame(ScriptExecutionContext& context) { return m_observer->takeVideoFrame(context); }

    private:
        VideoFrameObserverWrapper(ScriptExecutionContextIdentifier, MediaStreamTrackProcessor&, Ref<RealtimeMediaSource>&&, unsigned short maxVideoFramesCount);

        const UniqueRef<VideoFrameObserver> m_observer;
    };

    RefPtr<ReadableStream> m_readable;
    std::unique_ptr<Source> m_readableStreamSource;
    RefPtr<VideoFrameObserverWrapper> m_videoFrameObserverWrapper;
    const Ref<MediaStreamTrack> m_track;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && ENABLE(WEB_CODECS)
