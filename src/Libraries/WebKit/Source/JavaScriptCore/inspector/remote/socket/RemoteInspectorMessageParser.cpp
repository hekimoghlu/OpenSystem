/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "RemoteInspectorMessageParser.h"

#include <wtf/ByteOrder.h>

#if ENABLE(REMOTE_INSPECTOR)

namespace Inspector {

/*
| <--- one message for send / didReceiveData ---> |
+--------------+----------------------------------+--------------
|    size      |               data               | (next message)
|  4byte (NBO) |          variable length         |
+--------------+----------------------------------+--------------
               | <------------ size ------------> |
*/

MessageParser::MessageParser(Function<void(Vector<uint8_t>&&)>&& listener)
    : m_listener(WTFMove(listener))
{
}

Vector<uint8_t> MessageParser::createMessage(std::span<const uint8_t> data)
{
    if (data.empty() || data.size() > std::numeric_limits<uint32_t>::max())
        return Vector<uint8_t>();

    auto messageBuffer = Vector<uint8_t>(data.size() + sizeof(uint32_t));
    uint32_t nboSize = htonl(static_cast<uint32_t>(data.size()));
    memcpy(&messageBuffer[0], &nboSize, sizeof(uint32_t));
    memcpy(&messageBuffer[sizeof(uint32_t)], data.data(), data.size());
    return messageBuffer;
}

void MessageParser::pushReceivedData(std::span<const uint8_t> data)
{
    if (data.empty() || !m_listener)
        return;

    m_buffer.append(data);

    if (!parse())
        clearReceivedData();
}

void MessageParser::clearReceivedData()
{
    m_buffer.clear();
}

bool MessageParser::parse()
{
    while (m_buffer.size() >= sizeof(uint32_t)) {
        uint32_t dataSize;
        memcpy(&dataSize, &m_buffer[0], sizeof(uint32_t));
        dataSize = ntohl(dataSize);
        if (!dataSize) {
            LOG_ERROR("Message Parser received an invalid message size");
            return false;
        }

        size_t messageSize = (sizeof(uint32_t) + dataSize);
        if (m_buffer.size() < messageSize) {
            // Wait for more data.
            return true;
        }

        // FIXME: This should avoid re-creating a new data Vector.
        auto dataBuffer = Vector<uint8_t>(dataSize);
        memcpy(&dataBuffer[0], &m_buffer[sizeof(uint32_t)], dataSize);

        m_listener(WTFMove(dataBuffer));

        m_buffer.remove(0, messageSize);
    }

    return true;
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
