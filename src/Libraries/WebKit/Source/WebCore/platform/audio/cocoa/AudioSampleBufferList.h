/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include "CAAudioStreamDescription.h"
#include "CARingBuffer.h"
#include "WebAudioBufferList.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

typedef struct AudioStreamBasicDescription AudioStreamBasicDescription;
typedef struct OpaqueAudioConverter* AudioConverterRef;

namespace WebCore {

class AudioSampleBufferList : public RefCounted<AudioSampleBufferList> {
public:
    static Ref<AudioSampleBufferList> create(const CAAudioStreamDescription&, size_t);

    ~AudioSampleBufferList();

    static inline size_t audioBufferListSizeForStream(const CAAudioStreamDescription&);

    WEBCORE_EXPORT static void applyGain(AudioBufferList&, float, AudioStreamDescription::PCMFormat);
    void applyGain(float);

    OSStatus copyFrom(const AudioSampleBufferList&, size_t count = SIZE_MAX);
    OSStatus copyFrom(const AudioBufferList&, size_t frameCount, AudioConverterRef);
    OSStatus copyFrom(CARingBuffer&, size_t frameCount, uint64_t startFrame, CARingBuffer::FetchMode);

    OSStatus mixFrom(const AudioSampleBufferList&, size_t count = SIZE_MAX);

    OSStatus mixFrom(const AudioBufferList&, size_t count = SIZE_MAX);
    OSStatus copyTo(AudioBufferList&, size_t count = SIZE_MAX);

    const AudioStreamBasicDescription& streamDescription() const { return m_internalFormat.streamDescription(); }
    const WebAudioBufferList& bufferList() const { return m_bufferList; }
    WebAudioBufferList& bufferList() { return m_bufferList; }

    uint32_t sampleCapacity() const { return m_sampleCapacity; }
    uint32_t sampleCount() const { return m_sampleCount; }
    void setSampleCount(size_t);

    uint64_t timestamp() const { return m_timestamp; }
    double hostTime() const { return m_hostTime; }
    void setTimes(uint64_t time, double hostTime) { m_timestamp = time; m_hostTime = hostTime; }

    void reset();

    WEBCORE_EXPORT static void zeroABL(AudioBufferList&, size_t);
    void zero();

protected:
    AudioSampleBufferList(const CAAudioStreamDescription&, size_t);

    CAAudioStreamDescription m_internalFormat;

    uint64_t m_timestamp { 0 };
    double m_hostTime { -1 };
    size_t m_sampleCount { 0 };
    size_t m_sampleCapacity { 0 };
    size_t m_maxBufferSizePerChannel { 0 };
    size_t m_bufferListBaseSize { 0 };
    UniqueRef<WebAudioBufferList> m_bufferList;
};

inline size_t AudioSampleBufferList::audioBufferListSizeForStream(const CAAudioStreamDescription& description)
{
    return offsetof(AudioBufferList, mBuffers) + (sizeof(AudioBuffer) * std::max<uint32_t>(1, description.numberOfChannelStreams()));
}

} // namespace WebCore
