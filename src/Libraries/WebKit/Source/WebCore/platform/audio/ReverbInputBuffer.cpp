/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#include "ReverbInputBuffer.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ReverbInputBuffer);

ReverbInputBuffer::ReverbInputBuffer(size_t length)
    : m_buffer(length)
{
}

void ReverbInputBuffer::write(std::span<const float> source)
{
    size_t bufferLength = m_buffer.size();
    bool isCopySafe = m_writeIndex + source.size() <= bufferLength;
    ASSERT(isCopySafe);
    if (!isCopySafe)
        return;
        
    memcpySpan(m_buffer.span().subspan(m_writeIndex), source);

    m_writeIndex += source.size();
    ASSERT(m_writeIndex <= bufferLength);

    if (m_writeIndex >= bufferLength)
        m_writeIndex = 0;
}

std::span<float> ReverbInputBuffer::directReadFrom(int* readIndex, size_t numberOfFrames)
{
    size_t bufferLength = m_buffer.size();
    bool isPointerGood = readIndex && *readIndex >= 0 && *readIndex + numberOfFrames <= bufferLength;
    ASSERT(isPointerGood);
    if (!isPointerGood) {
        // Should never happen in practice but return pointer to start of buffer (avoid crash)
        if (readIndex)
            *readIndex = 0;
        return m_buffer.span().first(numberOfFrames);
    }
        
    auto source = m_buffer.span();
    auto p = source.subspan(*readIndex, numberOfFrames);

    // Update readIndex
    *readIndex = (*readIndex + numberOfFrames) % bufferLength;

    return p;
}

void ReverbInputBuffer::reset()
{
    m_buffer.zero();
    m_writeIndex = 0;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
