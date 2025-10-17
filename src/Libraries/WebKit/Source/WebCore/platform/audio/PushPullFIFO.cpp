/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

#if ENABLE(WEB_AUDIO)
#include "PushPullFIFO.h"

#include "AudioBus.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {

PushPullFIFO::PushPullFIFO(unsigned numberOfChannels, size_t fifoLength)
    : m_fifoLength(fifoLength)
{
    ASSERT(m_fifoLength <= maxFIFOLength);
    m_fifoBus = AudioBus::create(numberOfChannels, m_fifoLength);
}

PushPullFIFO::~PushPullFIFO() = default;

// Push the data from |inputBus| to FIFO. The size of push is determined by
// the length of |inputBus|.
void PushPullFIFO::push(const AudioBus* inputBus)
{
    ASSERT(inputBus);
    ASSERT(inputBus->length() <= m_fifoLength);
    ASSERT(m_indexWrite < m_fifoLength);

    const size_t inputBusLength = inputBus->length();
    const size_t remainder = m_fifoLength - m_indexWrite;

    for (unsigned i = 0; i < m_fifoBus->numberOfChannels(); ++i) {
        auto fifoBusChannel = m_fifoBus->channel(i)->mutableSpan();
        auto inputBusChannel = inputBus->channel(i)->span();
        if (remainder >= inputBusLength) {
            // The remainder is big enough for the input data.
            memcpySpan(fifoBusChannel.subspan(m_indexWrite), inputBusChannel);
        } else {
            // The input data overflows the remainder size. Wrap around the index.
            memcpySpan(fifoBusChannel.subspan(m_indexWrite), inputBusChannel.first(remainder));
            memcpySpan(fifoBusChannel, inputBusChannel.subspan(remainder));
        }
    }

    // Update the write index; wrap it around if necessary.
    m_indexWrite = (m_indexWrite + inputBusLength) % m_fifoLength;

    // In case of overflow, move the |indexRead| to the updated |indexWrite| to
    // avoid reading overwritten frames by the next pull.
    if (inputBusLength > m_fifoLength - m_framesAvailable)
        m_indexRead = m_indexWrite;

    // Update the number of frames available in FIFO.
    m_framesAvailable = std::min(m_framesAvailable + inputBusLength, m_fifoLength);
    ASSERT(((m_indexRead + m_framesAvailable) % m_fifoLength) == m_indexWrite);
}

// Pull the data out of FIFO to |outputBus|. If remaining frame in the FIFO
// is less than the frames to pull, provides remaining frame plus the silence.
size_t PushPullFIFO::pull(AudioBus* outputBus, size_t framesRequested)
{
    ASSERT(outputBus);
    ASSERT(framesRequested <= outputBus->length());
    ASSERT(framesRequested <= m_fifoLength);
    ASSERT(m_indexRead < m_fifoLength);

    const size_t remainder = m_fifoLength - m_indexRead;
    const size_t framesToFill = std::min(m_framesAvailable, framesRequested);

    for (unsigned i = 0; i < m_fifoBus->numberOfChannels(); ++i) {
        auto fifoBusChannel = m_fifoBus->channel(i)->span();
        auto outputBusChannel = outputBus->channel(i)->mutableSpan();

        // Fill up the output bus with the available frames first.
        if (remainder >= framesToFill) {
            // The remainder is big enough for the frames to pull.
            memcpySpan(outputBusChannel, fifoBusChannel.subspan(m_indexRead, framesToFill));
        } else {
            // The frames to pull is bigger than the remainder size.
            // Wrap around the index.
            memcpySpan(outputBusChannel, fifoBusChannel.subspan(m_indexRead, remainder));
            memcpySpan(outputBusChannel.subspan(remainder), fifoBusChannel.first(framesToFill - remainder));
        }

        // The frames available was not enough to fulfill the requested frames. Fill
        // the rest of the channel with silence.
        if (framesRequested > framesToFill)
            zeroSpan(outputBusChannel.subspan(framesToFill, framesRequested - framesToFill));
    }

    // Update the read index; wrap it around if necessary.
    m_indexRead = (m_indexRead + framesToFill) % m_fifoLength;

    // In case of underflow, move the |indexWrite| to the updated |indexRead|.
    if (framesRequested > framesToFill)
        m_indexWrite = m_indexRead;

    // Update the number of frames in FIFO.
    m_framesAvailable -= framesToFill;
    ASSERT(((m_indexRead + m_framesAvailable) % m_fifoLength) == m_indexWrite);

    // |framesRequested > m_framesAvailable| means the frames in FIFO is not
    // enough to fulfill the requested frames from the audio device.
    return framesRequested > m_framesAvailable ? framesRequested - m_framesAvailable : 0;
}

unsigned PushPullFIFO::numberOfChannels() const
{
    return m_fifoBus->numberOfChannels();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
