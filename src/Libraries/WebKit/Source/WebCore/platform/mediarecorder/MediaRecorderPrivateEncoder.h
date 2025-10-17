/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#include "CAAudioStreamDescription.h"
#include "MediaRecorderPrivateWriter.h"
#include "SharedBuffer.h"
#include "VideoEncoder.h"
#include <atomic>
#include <span>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/MediaTime.h>
#include <wtf/MonotonicTime.h>
#include <wtf/NativePromise.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WorkQueue.h>

typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;
typedef struct opaqueCMBufferQueueTriggerToken *CMBufferQueueTriggerToken;

namespace WebCore {

class AudioSampleBufferConverter;
class InProcessCARingBuffer;
class FragmentedSharedBuffer;
struct MediaRecorderPrivateOptions;
class MediaSample;
class PlatformAudioData;
class VideoFrame;
struct AudioInfo;
struct VideoInfo;

class MediaRecorderPrivateEncoder final : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaRecorderPrivateEncoder, WTF::DestructionThread::Main> {
public:
    static RefPtr<MediaRecorderPrivateEncoder> create(bool hasAudio, bool hasVideo, const MediaRecorderPrivateOptions&);
    ~MediaRecorderPrivateEncoder();

    void appendVideoFrame(VideoFrame&);
    void appendAudioSampleBuffer(const PlatformAudioData&, const AudioStreamDescription&, const WTF::MediaTime&, size_t);
    void fetchData(CompletionHandler<void(RefPtr<FragmentedSharedBuffer>&&, double)>&&);

    void pause();
    void resume();
    void stopRecording();
    void close();

    String mimeType() const;
    unsigned audioBitRate() const;
    unsigned videoBitRate() const;

    bool hasAudio() const { return m_hasAudio; }
    bool hasVideo() const { return m_hasVideo; }

private:
    MediaRecorderPrivateEncoder(bool hasAudio, bool hasVideo);
    bool initialize(const MediaRecorderPrivateOptions&, UniqueRef<MediaRecorderPrivateWriter>&&);

    static WorkQueue& queueSingleton();

    Ref<MediaRecorderPrivateWriterListener> listener();

    class Listener;
    friend class Listener;
    void appendData(std::span<const uint8_t>);

    MediaTime lastEnqueuedAudioTime() const { return MediaTime(m_lastEnqueuedAudioTimeUs.load(), 1000000); }
    MediaTime currentTime() const;
    MediaTime currentEndTime() const;

    void flushDataBuffer();

    static void compressedAudioOutputBufferCallback(void*, CMBufferQueueTriggerToken);

    Ref<FragmentedSharedBuffer> takeData();

    MediaTime lastMuxedSampleTime() const;

    void generateMIMEType();

    void audioSamplesDescriptionChanged(const AudioStreamBasicDescription&);
    void audioSamplesAvailable(const MediaTime&, size_t, size_t);
    RefPtr<AudioSampleBufferConverter> audioConverter() const;
    void enqueueCompressedAudioSampleBuffers();

    void appendVideoFrame(MediaTime, Ref<VideoFrame>&&);
    Ref<GenericPromise> encodePendingVideoFrames();
    void processVideoEncoderActiveConfiguration(const VideoEncoder::Config&, const VideoEncoderActiveConfiguration&);
    void enqueueCompressedVideoFrame(VideoEncoder::EncodedFrame&&);

    Ref<GenericPromise> flushPendingData(const MediaTime&);
    void partiallyFlushEncodedQueues();
    Ref<GenericPromise> waitForMatchingAudio(const MediaTime&);
    using Result = MediaRecorderPrivateWriter::Result;
    MediaTime flushToEndSegment(const MediaTime&);
    void flushAllEncodedQueues();
    void interleaveAndEnqueueNextFrame();

    void maybeStartWriter();
    bool hasMuxedDataSinceEndSegment() const;

    void addRingBuffer(const AudioStreamDescription&);
    void writeDataToRingBuffer(AudioBufferList*, size_t, size_t);
    void updateCurrentRingBufferIfNeeded();

    std::atomic<bool> m_isStopped { false };
    bool m_writerIsStarted WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_writerIsClosed WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    MediaTime m_lastMuxedSampleStartTime WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    bool m_lastMuxedSampleIsVideo WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    MediaTime m_lastMuxedAudioSampleEndTime WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    bool m_hasMuxedAudioFrameSinceEndSegment WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_hasMuxedVideoFrameSinceEndSegment WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_nextVideoFrameMuxedShouldBeKeyframe WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { true };
    size_t m_pendingFlush WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { 0 };
    Ref<GenericPromise> m_currentFlushOperations WTF_GUARDED_BY_CAPABILITY(mainThread) { GenericPromise::createAndResolve() };

    String m_mimeType WTF_GUARDED_BY_CAPABILITY(mainThread);

    FourCharCode m_audioCodec { 0 }; // set on creation. const after
    String m_audioCodecMimeType WTF_GUARDED_BY_CAPABILITY(mainThread);
    std::optional<uint8_t> m_audioTrackIndex WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RetainPtr<CMFormatDescriptionRef> m_audioFormatDescription WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RefPtr<AudioInfo> m_audioCompressedAudioInfo WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RefPtr<AudioSampleBufferConverter> m_audioConverter WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    bool m_formatChangedOccurred WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    std::optional<AudioStreamBasicDescription> m_originalOutputDescription WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    Deque<UniqueRef<MediaSamplesBlock>> m_encodedAudioFrames WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    std::optional<std::pair<const MediaTime, GenericPromise::Producer>> m_pendingAudioFramePromise WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    Lock m_ringBuffersLock;
    Deque<std::unique_ptr<InProcessCARingBuffer>> m_ringBuffers WTF_GUARDED_BY_LOCK(m_ringBuffersLock);
    InProcessCARingBuffer* m_currentRingBuffer WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { nullptr };
    std::atomic<int64_t> m_lastEnqueuedAudioTimeUs { 0 };
    std::atomic<int64_t> m_currentAudioTimeUs { 0 };

    // Audio thread variables.
    std::optional<CAAudioStreamDescription> m_currentStreamDescription;
    MediaTime m_currentAudioTime { MediaTime::zeroTime() };
    uint64_t m_currentAudioSampleCount { 0 };

    FourCharCode m_videoCodec { 0 }; // set on creation. const after
    String m_videoCodecMimeType WTF_GUARDED_BY_CAPABILITY(mainThread);
    std::optional<uint8_t> m_videoTrackIndex WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RefPtr<VideoInfo> m_videoTrackInfo WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RetainPtr<CMFormatDescriptionRef> m_videoFormatDescription WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    RefPtr<VideoEncoder> m_videoEncoder WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    Deque<std::pair<Ref<VideoFrame>, MediaTime>> m_pendingVideoFrames WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    Deque<UniqueRef<MediaSamplesBlock>> m_encodedVideoFrames WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    Deque<UniqueRef<MediaSamplesBlock>> m_interleavedFrames WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    bool m_firstVideoFrameProcessed WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    std::optional<MonotonicTime> m_currentVideoSegmentStartTime WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    uint64_t m_previousSegmentVideoDurationUs WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { 0 };
    MediaTime m_lastRawVideoFrameReceived WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { MediaTime::negativeInfiniteTime() };
    MediaTime m_lastEnqueuedRawVideoFrame WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { MediaTime::negativeInfiniteTime() };
    MediaTime m_lastVideoKeyframeTime WTF_GUARDED_BY_CAPABILITY(queueSingleton());
    std::optional<CGAffineTransform> m_videoTransform;
    RefPtr<GenericNonExclusivePromise> m_videoEncoderCreationPromise;

    Lock m_lock;
    SharedBufferBuilder m_data WTF_GUARDED_BY_LOCK(m_lock);
    static constexpr size_t s_dataBufferSize { 1024 };
    Vector<uint8_t> m_dataBuffer WTF_GUARDED_BY_LOCK(m_lock);
    double m_timeCode WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { 0 };

    uint64_t m_audioBitsPerSecond { 0 }; // set on creation. const after
    uint64_t m_videoBitsPerSecond { 0 }; // set on creation. const after

    bool m_hadError WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_isPaused WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_hasStartedAudibleAudioFrame WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { false };
    bool m_needKeyFrame WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { true };
    MediaTime m_startSegmentTime WTF_GUARDED_BY_CAPABILITY(queueSingleton()) { MediaTime::zeroTime() };

    const MediaTime m_minimumSegmentDuration { MediaTime::createWithDouble(1) };
    const MediaTime m_maxGOPDuration { MediaTime::createWithDouble(2) };
    const bool m_hasAudio { false };
    const bool m_hasVideo { false };
    const Ref<Listener> m_listener;
    std::unique_ptr<MediaRecorderPrivateWriter> m_writer; // Always set and immutable once initialize() has been called
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
