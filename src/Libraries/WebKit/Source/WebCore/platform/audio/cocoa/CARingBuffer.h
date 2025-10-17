/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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

#if ENABLE(WEB_AUDIO) && USE(MEDIATOOLBOX)

#include "AudioStreamDescription.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <optional>
#include <wtf/CheckedArithmetic.h>
#include <wtf/Lock.h>
#include <wtf/SequenceLocked.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>

typedef struct AudioBufferList AudioBufferList;

namespace WebCore {

class CAAudioStreamDescription;

class CARingBuffer {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CARingBuffer, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT virtual ~CARingBuffer();

    enum Error {
        Ok,
        TooMuch, // fetch start time is earlier than buffer start time and fetch end time is later than buffer end time
    };
    struct TimeBounds {
        uint64_t startFrame { 0 };
        uint64_t endFrame { 0 };
        bool operator<=>(const TimeBounds&) const = default;
    };
    WEBCORE_EXPORT TimeBounds getStoreTimeBounds();
    WEBCORE_EXPORT Error store(const AudioBufferList*, size_t frameCount, uint64_t startFrame);

    enum FetchMode { Copy, MixInt16, MixInt32, MixFloat32, MixFloat64 };
    static FetchMode fetchModeForMixing(AudioStreamDescription::PCMFormat);
    WEBCORE_EXPORT bool fetchIfHasEnoughData(AudioBufferList*, size_t frameCount, uint64_t startFrame, FetchMode = Copy);

    // Fills buffer with silence if there is not enough data.
    WEBCORE_EXPORT void fetch(AudioBufferList*, size_t frameCount, uint64_t startFrame, FetchMode = Copy);

    WEBCORE_EXPORT TimeBounds getFetchTimeBounds();

    uint32_t channelCount() const { return m_channelCount; }

protected:
    WEBCORE_EXPORT CARingBuffer(size_t bytesPerFrame, size_t frameCount, uint32_t numChannelStreams);
    WEBCORE_EXPORT void initialize();

    WEBCORE_EXPORT static CheckedSize computeCapacityBytes(size_t bytesPerFrame, size_t frameCount);
    WEBCORE_EXPORT static CheckedSize computeSizeForBuffers(size_t bytesPerFrame, size_t frameCount, uint32_t numChannelStreams);

    virtual void* data() = 0;
    std::span<uint8_t> span() { return unsafeMakeSpan(static_cast<uint8_t*>(data()), m_channelCount * m_capacityBytes); }
    using TimeBoundsBuffer = SequenceLocked<TimeBounds>;
    virtual TimeBoundsBuffer& timeBoundsBuffer() = 0;

private:
    size_t frameOffset(uint64_t frameNumber) const { return (frameNumber % m_frameCount) * m_bytesPerFrame; }
    void setTimeBounds(TimeBounds bufferBounds);
    void fetchInternal(AudioBufferList*, size_t frameCount, uint64_t startFrame, FetchMode, TimeBounds bufferBounds);

    Vector<std::span<Byte>> m_channels;
    const uint32_t m_channelCount;
    const size_t m_bytesPerFrame;
    const uint32_t m_frameCount;
    const size_t m_capacityBytes;

    // Stored range.
    TimeBounds m_storeBounds;
};

inline CARingBuffer::FetchMode CARingBuffer::fetchModeForMixing(AudioStreamDescription::PCMFormat format)
{
    switch (format) {
    case AudioStreamDescription::None:
    case AudioStreamDescription::Uint8:
    case AudioStreamDescription::Int24:
        ASSERT_NOT_REACHED();
        return MixInt32;
    case AudioStreamDescription::Int16:
        return MixInt16;
    case AudioStreamDescription::Int32:
        return MixInt32;
    case AudioStreamDescription::Float32:
        return MixFloat32;
    case AudioStreamDescription::Float64:
        return MixFloat64;
    }
}

class InProcessCARingBuffer final : public CARingBuffer {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(InProcessCARingBuffer, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static std::unique_ptr<InProcessCARingBuffer> allocate(const WebCore::CAAudioStreamDescription& format, size_t frameCount);
    WEBCORE_EXPORT ~InProcessCARingBuffer();

    TimeBoundsBuffer& timeBoundsBufferForTesting() { return timeBoundsBuffer(); }

protected:
    WEBCORE_EXPORT InProcessCARingBuffer(size_t bytesPerFrame, size_t frameCount, uint32_t numChannelStreams, Vector<uint8_t>&& buffer);
    void* data() final { return m_buffer.data(); }
    TimeBoundsBuffer& timeBoundsBuffer() final { return m_timeBoundsBuffer; }

private:
    Vector<uint8_t> m_buffer;
    TimeBoundsBuffer m_timeBoundsBuffer;
};

}

#endif // ENABLE(WEB_AUDIO) && USE(MEDIATOOLBOX)
