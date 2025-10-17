/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#include "HTTPParser.h"

#include <wtf/text/StringToIntegerConversion.h>

namespace WebDriver {

HTTPParser::Phase HTTPParser::parse(Vector<uint8_t>&& data)
{
    if (!data.isEmpty()) {
        m_buffer.appendVector(WTFMove(data));

        while (true) {
            if (handlePhase() == Process::Suspend)
                break;
        }
    }

    return m_phase;
}

HTTPParser::Process HTTPParser::handlePhase()
{
    switch (m_phase) {
    case Phase::Idle: {
        String line;
        if (!readLine(line))
            return Process::Suspend;

        if (!parseFirstLine(WTFMove(line)))
            return abortProcess("Client error: invalid request line.");

        ASSERT(!m_message.method.isEmpty());
        ASSERT(!m_message.path.isEmpty());
        ASSERT(!m_message.version.isEmpty());
        m_phase = Phase::Header;

        return Process::Continue;
    }

    case Phase::Header: {
        String line;
        if (!readLine(line))
            return Process::Suspend;

        if (!line.isEmpty())
            m_message.requestHeaders.append(WTFMove(line));
        else {
            m_bodyLength = expectedBodyLength();
            m_phase = Phase::Body;
        }

        return Process::Continue;
    }

    case Phase::Body:
        if (m_buffer.size() > m_bodyLength)
            return abortProcess("Client error: don't send data after request and before response.");

        if (m_buffer.size() < m_bodyLength)
            return Process::Suspend;

        m_message.requestBody = WTFMove(m_buffer);
        m_phase = Phase::Complete;

        return Process::Suspend;

    case Phase::Complete:
        return abortProcess("Client error: don't send data after request and before response.");

    case Phase::Error:
        return abortProcess();
    }
}

HTTPParser::Process HTTPParser::abortProcess(const char* message)
{
    if (message)
        LOG_ERROR("%s", message);

    m_phase = Phase::Error;

    m_buffer.shrink(0);

    return Process::Suspend;
}

bool HTTPParser::parseFirstLine(String&& line)
{
    auto components = line.split(' ');
    if (components.size() != 3)
        return false;

    m_message.method = WTFMove(components[0]);
    m_message.path = WTFMove(components[1]);
    m_message.version = WTFMove(components[2]);
    return true;
}

bool HTTPParser::readLine(String& line)
{
    auto length = m_buffer.size();
    auto position = m_buffer.find(0x0d);
    if (position == notFound || position + 1 == length || m_buffer[position + 1] != 0x0a)
        return false;

    line = String::fromUTF8({ m_buffer.data(), position });
    if (line.isNull())
        LOG_ERROR("Client error: invalid encoding in HTTP header.");

    m_buffer.remove(0, position + 2);
    return true;
}

size_t HTTPParser::expectedBodyLength() const
{
    if (m_message.method == "HEAD"_s)
        return 0;

    constexpr auto name = "content-length:"_s;
    const size_t nameLength = name.length();

    for (const auto& header : m_message.requestHeaders) {
        if (header.startsWithIgnoringASCIICase(name))
            return parseIntegerAllowingTrailingJunk<size_t>(StringView { header }.substring(nameLength)).value_or(0);
    }

    return 0;
}

} // namespace WebDriver
