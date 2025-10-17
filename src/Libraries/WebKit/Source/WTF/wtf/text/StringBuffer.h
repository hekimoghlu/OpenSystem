/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#include <limits>
#include <unicode/utypes.h>
#include <wtf/Assertions.h>
#include <wtf/DebugHeap.h>
#include <wtf/MallocSpan.h>
#include <wtf/Noncopyable.h>

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StringBuffer);

template <typename CharType>
class StringBuffer {
    WTF_MAKE_NONCOPYABLE(StringBuffer);
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit StringBuffer(unsigned length)
        : m_length(length)
        , m_data(m_length ? static_cast<CharType*>(StringBufferMalloc::malloc(Checked<size_t>(m_length) * sizeof(CharType))) : nullptr)
    {
    }

    ~StringBuffer()
    {
        StringBufferMalloc::free(m_data);
    }

    void shrink(unsigned newLength)
    {
        ASSERT(newLength <= m_length);
        m_length = newLength;
    }

    void resize(unsigned newLength)
    {
        if (newLength > m_length)
            m_data = static_cast<CharType*>(StringBufferMalloc::realloc(m_data, Checked<size_t>(newLength) * sizeof(CharType)));
        m_length = newLength;
    }

    unsigned length() const { return m_length; }
    CharType* characters() { return m_data; }
    std::span<CharType> span() { return unsafeMakeSpan(m_data, m_length); }

    CharType& operator[](unsigned i) { RELEASE_ASSERT(i < m_length); return m_data[i]; }

    MallocSpan<CharType, StringBufferMalloc> release()
    {
        return adoptMallocSpan<CharType, StringBufferMalloc>(unsafeMakeSpan(std::exchange(m_data, nullptr), std::exchange(m_length, 0)));
    }

private:
    unsigned m_length;
    CharType* m_data;
};

} // namespace WTF

using WTF::StringBuffer;
