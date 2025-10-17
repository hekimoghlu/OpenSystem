/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#include <wtf/Deque.h>
#include <wtf/text/ParsingUtilities.h>

namespace WTF {

template <typename T, size_t BlockSize>
class StreamBuffer {
    WTF_MAKE_FAST_ALLOCATED;
private:
    typedef Vector<T> Block;
public:
    StreamBuffer()
        : m_size(0)
        , m_readOffset(0)
    {
    }

    ~StreamBuffer()
    {
    }

    bool isEmpty() const { return !size(); }

    void append(std::span<const T> data)
    {
        if (!data.size())
            return;

        m_size += data.size();
        while (data.size()) {
            if (!m_buffer.size() || m_buffer.last()->size() == BlockSize)
                m_buffer.append(makeUnique<Block>());
            size_t appendSize = std::min(BlockSize - m_buffer.last()->size(), data.size());
            m_buffer.last()->append(consumeSpan(data, appendSize));
        }
    }

    void append(const T* data, size_t size) { append(std::span { data, size }); }

    // This function consume data in the fist block.
    // Specified size must be less than over equal to firstBlockSize().
    void consume(size_t size)
    {
        ASSERT(m_size >= size);
        if (!m_size)
            return;

        ASSERT(m_buffer.size() > 0);
        ASSERT(m_readOffset + size <= m_buffer.first()->size());
        m_readOffset += size;
        m_size -= size;
        if (m_readOffset >= m_buffer.first()->size()) {
            m_readOffset = 0;
            m_buffer.removeFirst();
        }
    }

    size_t size() const { return m_size; }

    const T* firstBlockData() const
    {
        if (!m_size)
            return 0;
        ASSERT(m_buffer.size() > 0);
        return &m_buffer.first()->data()[m_readOffset];
    }

    size_t firstBlockSize() const
    {
        if (!m_size)
            return 0;
        ASSERT(m_buffer.size() > 0);
        return m_buffer.first()->size() - m_readOffset;
    }

    std::span<const T> firstBlockSpan() const { return std::span { firstBlockData(), firstBlockSize() }; }

private:
    size_t m_size;
    size_t m_readOffset;
    Deque<std::unique_ptr<Block>> m_buffer;
};

} // namespace WTF

using WTF::StreamBuffer;
