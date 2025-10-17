/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#include "AudioSampleDataConverter.h"
#include "CAAudioStreamDescription.h"
#include "CARingBuffer.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

typedef struct OpaqueAudioConverter* AudioConverterRef;
typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;

namespace WebCore {

class AudioSampleBufferList;
class PlatformAudioData;

class AudioSampleDataSource : public ThreadSafeRefCounted<AudioSampleDataSource, WTF::DestructionThread::MainRunLoop>
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
    {
public:
    static Ref<AudioSampleDataSource> create(size_t, LoggerHelper&, size_t waitToStartForPushCount = 2);

    ~AudioSampleDataSource();

    OSStatus setInputFormat(const CAAudioStreamDescription&);
    OSStatus setOutputFormat(const CAAudioStreamDescription&);

    void pushSamples(const MediaTime&, const PlatformAudioData&, size_t);
    void pushSamples(const AudioStreamBasicDescription&, CMSampleBufferRef);

    enum PullMode { Copy, Mix };
    bool pullSamples(AudioBufferList&, size_t, uint64_t, double, PullMode);

    bool pullAvailableSamplesAsChunks(AudioBufferList&, size_t frameCount, uint64_t timeStamp, Function<void()>&&);
    bool pullAvailableSampleChunk(AudioBufferList&, size_t frameCount, uint64_t timeStamp, PullMode);

    void setVolume(float volume) { m_volume = volume; }
    float volume() const { return m_volume; }

    void setMuted(bool muted) { m_muted = muted; }
    bool muted() const { return m_muted; }

    const CAAudioStreamDescription* inputDescription() const { return m_inputDescription ? &m_inputDescription.value() : nullptr; }
    const CAAudioStreamDescription* outputDescription() const { return m_outputDescription ? &m_outputDescription.value() : nullptr; }

    void recomputeSampleOffset() { m_shouldComputeOutputSampleOffset = true; }

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    void setLogger(Ref<const Logger>&&, uint64_t);
#endif

    static constexpr float EquivalentToMaxVolume = 0.95;

private:
    AudioSampleDataSource(size_t, LoggerHelper&, size_t waitToStartForPushCount);

    OSStatus setupConverter();

    void pushSamplesInternal(const AudioBufferList&, const MediaTime&, size_t frameCount);
    bool pullSamplesInternal(AudioBufferList&, size_t sampleCount, uint64_t timeStamp, PullMode);

    std::optional<CAAudioStreamDescription> m_inputDescription;
    std::optional<CAAudioStreamDescription> m_outputDescription;

    MediaTime hostTime() const;

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "AudioSampleDataSource"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    uint64_t m_lastPushedSampleCount { 0 };
    size_t m_waitToStartForPushCount { 2 };

    int64_t m_expectedNextPushedSampleTimeValue { 0 };
    int64_t m_converterInputOffset { 0 };
    std::optional<int64_t> m_inputSampleOffset;
    int64_t m_outputSampleOffset { 0 };
    uint64_t m_lastBufferedAmount { 0 };

    AudioSampleDataConverter m_converter;

    RefPtr<AudioSampleBufferList> m_scratchBuffer;

    std::unique_ptr<InProcessCARingBuffer> m_ringBuffer;
    size_t m_maximumSampleCount { 0 };

    float m_volume { 1.0 };
    bool m_muted { false };
    bool m_shouldComputeOutputSampleOffset { true };

    bool m_isInNeedOfMoreData { false };
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore
