/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#include "SharedBufferChunkReader.h"

#include <wtf/text/StringCommon.h>

namespace WebCore {

#if ENABLE(MHTML)

SharedBufferChunkReader::SharedBufferChunkReader(FragmentedSharedBuffer* buffer, const Vector<char>& separator)
    : m_iteratorCurrent(buffer->begin())
    , m_iteratorEnd(buffer->end())
    , m_segment(m_iteratorCurrent != m_iteratorEnd ? m_iteratorCurrent->segment->span().data() : nullptr)
    , m_separator(separator)
{
}

SharedBufferChunkReader::SharedBufferChunkReader(FragmentedSharedBuffer* buffer, const char* separator)
    : m_iteratorCurrent(buffer->begin())
    , m_iteratorEnd(buffer->end())
    , m_segment(m_iteratorCurrent != m_iteratorEnd ? m_iteratorCurrent->segment->span().data() : nullptr)
{
    setSeparator(separator);
}

void SharedBufferChunkReader::setSeparator(const Vector<char>& separator)
{
    m_separator = separator;
}

void SharedBufferChunkReader::setSeparator(const char* separator)
{
    m_separator.clear();
    m_separator.append(unsafeSpan(separator));
}

bool SharedBufferChunkReader::nextChunk(Vector<uint8_t>& chunk, bool includeSeparator)
{
    if (m_iteratorCurrent == m_iteratorEnd)
        return false;

    chunk.clear();
    while (true) {
        while (m_segmentIndex < m_iteratorCurrent->segment->size()) {
            // FIXME: The existing code to check for separators doesn't work correctly with arbitrary separator strings.
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
            auto currentCharacter = m_segment[m_segmentIndex++];
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
            if (currentCharacter != m_separator[m_separatorIndex]) {
                if (m_separatorIndex > 0) {
                    ASSERT_WITH_SECURITY_IMPLICATION(m_separatorIndex <= m_separator.size());
                    chunk.append(m_separator.span().first(m_separatorIndex));
                    m_separatorIndex = 0;
                }
                chunk.append(currentCharacter);
                continue;
            }
            m_separatorIndex++;
            if (m_separatorIndex == m_separator.size()) {
                if (includeSeparator)
                    chunk.appendVector(m_separator);
                m_separatorIndex = 0;
                return true;
            }
        }

        // Read the next segment.
        m_segmentIndex = 0;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (++m_iteratorCurrent == m_iteratorEnd) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
            m_segment = nullptr;
            if (m_separatorIndex > 0)
                chunk.append(byteCast<uint8_t>(m_separator.subspan(0, m_separatorIndex)));
            return !chunk.isEmpty();
        }
        m_segment = m_iteratorCurrent->segment->span().data();
    }

    ASSERT_NOT_REACHED();
    return false;
}

String SharedBufferChunkReader::nextChunkAsUTF8StringWithLatin1Fallback(bool includeSeparator)
{
    Vector<uint8_t> data;
    if (!nextChunk(data, includeSeparator))
        return String();

    return data.size() ? String::fromUTF8WithLatin1Fallback(data.span()) : emptyString();
}

size_t SharedBufferChunkReader::peek(Vector<uint8_t>& data, size_t requestedSize)
{
    data.clear();
    if (m_iteratorCurrent == m_iteratorEnd)
        return 0;

    size_t availableInSegment = std::min(m_iteratorCurrent->segment->size() - m_segmentIndex, requestedSize);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    data.append(unsafeMakeSpan(m_segment + m_segmentIndex, availableInSegment));

    size_t readBytesCount = availableInSegment;
    requestedSize -= readBytesCount;

    auto currentSegment = m_iteratorCurrent;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    while (requestedSize && ++currentSegment != m_iteratorEnd) {
        size_t lengthInSegment = std::min(currentSegment->segment->size(), requestedSize);
        data.append(currentSegment->segment->span().first(lengthInSegment));
        readBytesCount += lengthInSegment;
        requestedSize -= lengthInSegment;
    }
    return readBytesCount;
}

#endif

}

