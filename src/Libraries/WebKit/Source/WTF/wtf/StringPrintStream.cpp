/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include <wtf/StringPrintStream.h>

#include <stdarg.h>
#include <stdio.h>
#include <wtf/MallocSpan.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

StringPrintStream::StringPrintStream()
{
    m_buffer = std::span { m_inlineBuffer };
    m_buffer[0] = 0; // Make sure that we always have a null terminator.
}

StringPrintStream::~StringPrintStream()
{
    if (m_buffer.data() != m_inlineBuffer.data())
        fastFree(m_buffer.data());
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

void StringPrintStream::vprintf(const char* format, va_list argList)
{
    ASSERT_WITH_SECURITY_IMPLICATION(m_length < m_buffer.size());
    ASSERT(!m_buffer[m_length]);

    va_list firstPassArgList;
    va_copy(firstPassArgList, argList);

    int numberOfBytesNotIncludingTerminatorThatWouldHaveBeenWritten =
        vsnprintf(m_buffer.subspan(m_length).data(), m_buffer.size() - m_length, format, firstPassArgList);

    va_end(firstPassArgList);

    int numberOfBytesThatWouldHaveBeenWritten =
        numberOfBytesNotIncludingTerminatorThatWouldHaveBeenWritten + 1;

    if (m_length + numberOfBytesThatWouldHaveBeenWritten <= m_buffer.size()) {
        m_length += numberOfBytesNotIncludingTerminatorThatWouldHaveBeenWritten;
        return; // This means that vsnprintf() succeeded.
    }

    increaseSize(m_length + numberOfBytesThatWouldHaveBeenWritten);

    int numberOfBytesNotIncludingTerminatorThatWereWritten =
        vsnprintf(m_buffer.subspan(m_length).data(), m_buffer.size() - m_length, format, argList);

    int numberOfBytesThatWereWritten = numberOfBytesNotIncludingTerminatorThatWereWritten + 1;

    ASSERT_UNUSED(numberOfBytesThatWereWritten, m_length + numberOfBytesThatWereWritten <= m_buffer.size());

    m_length += numberOfBytesNotIncludingTerminatorThatWereWritten;

    ASSERT_WITH_SECURITY_IMPLICATION(m_length < m_buffer.size());
    ASSERT(!m_buffer[m_length]);
}

CString StringPrintStream::toCString() const
{
    ASSERT(m_length == strlenSpan(m_buffer));
    return CString(m_buffer.first(m_length));
}

void StringPrintStream::reset()
{
    m_length = 0;
    m_buffer[0] = 0;
}

Expected<String, UTF8ConversionError> StringPrintStream::tryToString() const
{
    ASSERT(m_length == strlenSpan(m_buffer));
    if (m_length > String::MaxLength)
        return makeUnexpected(UTF8ConversionError::OutOfMemory);
    return String::fromUTF8(m_buffer.first(m_length));
}

String StringPrintStream::toString() const
{
    ASSERT(m_length == strlenSpan(m_buffer));
    return String::fromUTF8(m_buffer.first(m_length));
}

String StringPrintStream::toStringWithLatin1Fallback() const
{
    ASSERT(m_length == strlenSpan(m_buffer));
    return String::fromUTF8WithLatin1Fallback(m_buffer.first(m_length));
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

void StringPrintStream::increaseSize(size_t newSize)
{
    ASSERT_WITH_SECURITY_IMPLICATION(newSize > m_buffer.size());
    ASSERT(newSize > m_inlineBuffer.size());

    // Use exponential resizing to reduce thrashing.
    newSize = newSize << 1;

    // Use fastMalloc instead of fastRealloc because we know that for the sizes we're using,
    // fastRealloc will just do malloc+free anyway. Also, this simplifies the code since
    // we can't realloc the inline buffer.
    auto newBuffer = MallocSpan<char>::malloc(newSize);
    memcpySpan(newBuffer.mutableSpan(), m_buffer.first(m_length + 1));
    if (m_buffer.data() != m_inlineBuffer.data())
        fastFree(m_buffer.data());
    m_buffer = newBuffer.leakSpan();
}

} // namespace WTF
