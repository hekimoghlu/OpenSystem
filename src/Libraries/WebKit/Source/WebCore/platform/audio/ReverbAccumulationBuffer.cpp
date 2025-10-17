/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

#include "ReverbAccumulationBuffer.h"

#include "VectorMath.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#include <algorithm>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ReverbAccumulationBuffer);

ReverbAccumulationBuffer::ReverbAccumulationBuffer(size_t length)
    : m_buffer(length)
{
}

void ReverbAccumulationBuffer::readAndClear(std::span<float> destination, size_t numberOfFrames)
{
    size_t bufferLength = m_buffer.size();
    bool isCopySafe = m_readIndex <= bufferLength && numberOfFrames <= bufferLength;
    
    ASSERT(isCopySafe);
    if (!isCopySafe)
        return;

    size_t framesAvailable = bufferLength - m_readIndex;
    size_t numberOfFrames1 = std::min(numberOfFrames, framesAvailable);
    size_t numberOfFrames2 = numberOfFrames - numberOfFrames1;

    auto source = m_buffer.span();
    memcpySpan(destination, source.subspan(m_readIndex).first(numberOfFrames1));
    zeroSpan(source.subspan(m_readIndex, numberOfFrames1));

    // Handle wrap-around if necessary
    if (numberOfFrames2 > 0) {
        memcpySpan(destination.subspan(numberOfFrames1), source.first(numberOfFrames2));
        zeroSpan(source.first(numberOfFrames2));
    }

    m_readIndex = (m_readIndex + numberOfFrames) % bufferLength;
    m_readTimeFrame += numberOfFrames;
}

void ReverbAccumulationBuffer::updateReadIndex(int* readIndex, size_t numberOfFrames) const
{
    // Update caller's readIndex
    *readIndex = (*readIndex + numberOfFrames) % m_buffer.size();
}

int ReverbAccumulationBuffer::accumulate(std::span<float> source, size_t numberOfFrames, int* readIndex, size_t delayFrames)
{
    size_t bufferLength = m_buffer.size();
    
    size_t writeIndex = (*readIndex + delayFrames) % bufferLength;

    // Update caller's readIndex
    *readIndex = (*readIndex + numberOfFrames) % bufferLength;

    size_t framesAvailable = bufferLength - writeIndex;
    size_t numberOfFrames1 = std::min(numberOfFrames, framesAvailable);
    size_t numberOfFrames2 = numberOfFrames - numberOfFrames1;

    auto destination = m_buffer.span();

    bool isSafe = writeIndex <= bufferLength && numberOfFrames1 + writeIndex <= bufferLength && numberOfFrames2 <= bufferLength;
    ASSERT(isSafe);
    if (!isSafe)
        return 0;

    VectorMath::add(source.first(numberOfFrames1), destination.subspan(writeIndex).first(numberOfFrames1), destination.subspan(writeIndex));

    // Handle wrap-around if necessary
    if (numberOfFrames2 > 0)       
        VectorMath::add(source.subspan(numberOfFrames1).first(numberOfFrames2), destination.first(numberOfFrames2), destination);

    return writeIndex;
}

void ReverbAccumulationBuffer::reset()
{
    m_buffer.zero();
    m_readIndex = 0;
    m_readTimeFrame = 0;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
