/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

#if USE(GSTREAMER_TRANSCODER)

#include "GRefPtrGStreamer.h"
#include "MediaRecorderPrivate.h"
#include "SharedBuffer.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Condition.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ContentType;
class MediaStreamTrackPrivate;
struct MediaRecorderPrivateOptions;

class MediaRecorderPrivateBackend : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaRecorderPrivateBackend, WTF::DestructionThread::Main> {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateBackend);
public:
    using SelectTracksCallback = Function<void(MediaRecorderPrivate::AudioVideoSelectedTracks)>;
    static RefPtr<MediaRecorderPrivateBackend> create(MediaStreamPrivate& stream, const MediaRecorderPrivateOptions& options)
    {
        return adoptRef(*new MediaRecorderPrivateBackend(stream, options));
    }

    ~MediaRecorderPrivateBackend();

    bool preparePipeline();

    void fetchData(MediaRecorderPrivate::FetchDataCallback&&);
    void startRecording(MediaRecorderPrivate::StartRecordingCallback&&);
    void stopRecording(CompletionHandler<void()>&&);
    void pauseRecording(CompletionHandler<void()>&&);
    void resumeRecording(CompletionHandler<void()>&&);
    const String& mimeType() const { return m_mimeType; }

    void setSelectTracksCallback(SelectTracksCallback&& callback) { m_selectTracksCallback = WTFMove(callback); }

private:
    MediaRecorderPrivateBackend(MediaStreamPrivate&, const MediaRecorderPrivateOptions&);

    void setSource(GstElement*);
    void setSink(GstElement*);
    void configureAudioEncoder(GstElement*);
    void configureVideoEncoder(GstElement*);

    GRefPtr<GstEncodingContainerProfile> containerProfile();
    MediaStreamPrivate& stream() const { return m_stream; }
    void processSample(GRefPtr<GstSample>&&);
    void notifyPosition(GstClockTime);
    void notifyEOS();

    GRefPtr<GstEncodingProfile> m_audioEncodingProfile;
    GRefPtr<GstEncodingProfile> m_videoEncodingProfile;
    String m_videoCodec;
    GRefPtr<GstTranscoder> m_transcoder;
    GRefPtr<GstTranscoderSignalAdapter> m_signalAdapter;
    GRefPtr<GstElement> m_pipeline;
    GRefPtr<GstElement> m_src;
    GRefPtr<GstElement> m_sink;
    Condition m_eosCondition;
    Lock m_eosLock;
    bool m_eos WTF_GUARDED_BY_LOCK(m_eosLock);

    Lock m_dataLock;
    SharedBufferBuilder m_data WTF_GUARDED_BY_LOCK(m_dataLock);
    MediaTime m_position WTF_GUARDED_BY_LOCK(m_dataLock) { MediaTime::invalidTime() };
    double m_timeCode WTF_GUARDED_BY_LOCK(m_dataLock) { 0 };

    MediaStreamPrivate& m_stream;
    const MediaRecorderPrivateOptions& m_options;
    String m_mimeType;
    std::optional<SelectTracksCallback> m_selectTracksCallback;
};

class MediaRecorderPrivateGStreamer final : public MediaRecorderPrivate {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateGStreamer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaRecorderPrivateGStreamer);
public:
    static std::unique_ptr<MediaRecorderPrivateGStreamer> create(MediaStreamPrivate&, const MediaRecorderPrivateOptions&);
    explicit MediaRecorderPrivateGStreamer(Ref<MediaRecorderPrivateBackend>&&);
    ~MediaRecorderPrivateGStreamer() = default;

    static bool isTypeSupported(const ContentType&);

private:
    void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) final { };
    void audioSamplesAvailable(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final { };

    void fetchData(FetchDataCallback&&) final;
    void startRecording(StartRecordingCallback&&) final;
    void stopRecording(CompletionHandler<void()>&&) final;
    void pauseRecording(CompletionHandler<void()>&&) final;
    void resumeRecording(CompletionHandler<void()>&&) final;
    String mimeType() const final;

    Ref<MediaRecorderPrivateBackend> m_recorder;
};

} // namespace WebCore

#endif // USE(GSTREAMER_TRANSCODER)
