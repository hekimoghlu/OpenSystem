/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#include "RealtimeMediaSource.h"
#include "WritableStreamSink.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class MediaStreamTrack;
class ScriptExecutionContext;
class WritableStream;

class VideoTrackGenerator : public RefCounted<VideoTrackGenerator> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(VideoTrackGenerator);
public:
    static ExceptionOr<Ref<VideoTrackGenerator>> create(ScriptExecutionContext&);
    ~VideoTrackGenerator();

    void setMuted(ScriptExecutionContext&, bool);
    bool muted(ScriptExecutionContext&) const { return m_muted; }

    Ref<WritableStream> writable();
    Ref<MediaStreamTrack> track();

private:
    class Sink;
    VideoTrackGenerator(Ref<Sink>&&, Ref<WritableStream>&&, Ref<MediaStreamTrack>&&);

    class Source final : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<Source, WTF::DestructionThread::MainRunLoop> {
    public:
        static Ref<Source> create(ScriptExecutionContextIdentifier identifier) { return adoptRef(*new Source(identifier)); }

        WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

        void writeVideoFrame(VideoFrame&, VideoFrameTimeMetadata);

        void setWritable(WritableStream&);

    private:
        explicit Source(ScriptExecutionContextIdentifier);

        const RealtimeMediaSourceCapabilities& capabilities() final { return m_capabilities; }
        const RealtimeMediaSourceSettings& settings() final { return m_settings; }
        void endProducingData() final;

        ScriptExecutionContextIdentifier m_contextIdentifier;
        WeakPtr<WritableStream> m_writable;

        RealtimeMediaSourceCapabilities m_capabilities;
        RealtimeMediaSourceSettings m_settings;
        IntSize m_videoFrameSize;
        IntSize m_maxVideoFrameSize;
    };

    class Sink final : public WritableStreamSink {
    public:
        static Ref<Sink> create(Ref<Source>&& source) { return adoptRef(*new Sink(WTFMove(source))); }

        void setMuted(bool muted) { m_muted = muted; }

    private:
        explicit Sink(Ref<Source>&&);

        void write(ScriptExecutionContext&, JSC::JSValue, DOMPromiseDeferred<void>&&) final;
        void close() final;
        void error(String&&) final;

        bool m_muted { false };
        Ref<Source> m_source;
    };

    bool m_muted { false };
    bool m_hasMutedChanged { false };
    Ref<Sink> m_sink;
    Ref<WritableStream> m_writable;
    Ref<MediaStreamTrack> m_track;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && ENABLE(WEB_CODECS)
