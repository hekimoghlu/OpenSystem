/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#ifndef AudioChannel_h
#define AudioChannel_h

#include "AudioArray.h"
#include <memory>
#include <span>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// An AudioChannel represents a buffer of non-interleaved floating-point audio samples.
// The PCM samples are normally assumed to be in a nominal range -1.0 -> +1.0
class AudioChannel final {
    WTF_MAKE_TZONE_ALLOCATED(AudioChannel);
    WTF_MAKE_NONCOPYABLE(AudioChannel);
public:
    // Memory can be externally referenced, or can be internally allocated with an AudioFloatArray.

    // Reference an external buffer.
    AudioChannel(std::span<float> storage)
        : m_span(storage)
        , m_silent(false)
    {
    }

    // Manage storage for us.
    explicit AudioChannel(size_t length)
        : m_memBuffer(makeUnique<AudioFloatArray>(length))
        , m_span(m_memBuffer->span())
    {
    }

    // A "blank" audio channel -- must call set() before it's useful...
    AudioChannel() = default;

    // Redefine the memory for this channel.
    // storage represents external memory not managed by this object.
    void set(std::span<float> storage)
    {
        m_memBuffer = nullptr; // cleanup managed storage
        m_span = storage;
        m_silent = false;
    }

    // How many sample-frames do we contain?
    size_t length() const { return m_span.size(); }

    // Set new length. Can only be set to a value lower than the current length.
    void setLength(size_t newLength)
    {
        m_span = m_span.first(newLength);
    }

    std::span<const float> span() const { return m_span; }
    std::span<float> mutableSpan()
    {
        clearSilentFlag();
        return m_span;
    }

    // Direct access to PCM sample data. Non-const accessor clears silent flag.
    float* mutableData()
    {
        clearSilentFlag();
        return m_span.data();
    }

    const float* data() const { return m_span.data(); }

    // Zeroes out all sample values in buffer.
    void zero()
    {
        if (m_silent)
            return;

        m_silent = true;
        if (m_memBuffer)
            m_memBuffer->zero();
        else
            zeroSpan(m_span);
    }

    // Clears the silent flag.
    void clearSilentFlag() { m_silent = false; }

    bool isSilent() const { return m_silent; }

    // Scales all samples by the same amount.
    void scale(float scale);

    // A simple memcpySpan() from the source channel
    void copyFrom(const AudioChannel* sourceChannel);

    // Copies the given range from the source channel.
    void copyFromRange(const AudioChannel* sourceChannel, unsigned startFrame, unsigned endFrame);

    // Sums (with unity gain) from the source channel.
    void sumFrom(const AudioChannel* sourceChannel);

    // Returns maximum absolute value (useful for normalization).
    float maxAbsValue() const;

private:
    std::unique_ptr<AudioFloatArray> m_memBuffer;
    std::span<float> m_span;
    bool m_silent { true };
};

} // WebCore

#endif // AudioChannel_h
