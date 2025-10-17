/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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
#include "OSLogPrintStream.h"

#include <wtf/StdLibExtras.h>
#include <wtf/text/ParsingUtilities.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

#if OS(DARWIN)

OSLogPrintStream::OSLogPrintStream(os_log_t log, os_log_type_t logType)
    : m_log(log)
    , m_logType(logType)
    , m_string("initial string... lol")
{
}

OSLogPrintStream::~OSLogPrintStream()
{ }

std::unique_ptr<OSLogPrintStream> OSLogPrintStream::open(const char* subsystem, const char* category, os_log_type_t logType)
{
    os_log_t log = os_log_create(subsystem, category);
    return makeUnique<OSLogPrintStream>(log, logType);
}

void OSLogPrintStream::vprintf(const char* format, va_list argList)
{
    Locker lock { m_stringLock };
    size_t offset = m_offset;
    size_t freeBytes = m_string.length() - offset;
    va_list backup;
    va_copy(backup, argList);
ALLOW_NONLITERAL_FORMAT_BEGIN
    size_t bytesWritten = vsnprintf(m_string.mutableSpanIncludingNullTerminator().subspan(offset).data(), freeBytes, format, argList);
    if (UNLIKELY(bytesWritten >= freeBytes)) {
        size_t newLength = std::max(bytesWritten + m_string.length(), m_string.length() * 2);
        m_string.grow(newLength);
        freeBytes = newLength - offset;
        bytesWritten = vsnprintf(m_string.mutableSpanIncludingNullTerminator().subspan(offset).data(), freeBytes, format, backup);
        ASSERT(bytesWritten < freeBytes);
    }
ALLOW_NONLITERAL_FORMAT_END

    size_t newOffset = offset + bytesWritten;
    auto buffer = m_string.mutableSpan();
    bool loggedText = false;
    do {
        if (buffer[offset] == '\n') {
            // Set the new line to a null character so os_log stops copying there.
            buffer[offset] = '\0';
            os_log_with_type(m_log, m_logType, "%{public}s", buffer.data());
            skip(buffer, offset + 1);
            newOffset -= offset + 1;
            offset = 0;
            loggedText = true;
        } else
            offset++;
    } while (offset < newOffset);

    if (loggedText)
        memmoveSpan(m_string.mutableSpan(), buffer.first(newOffset));
    m_offset = newOffset;
}

#endif

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
