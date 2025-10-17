/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "STUNMessageParsing.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include <LibWebRTCMacros.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/ParsingUtilities.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/byte_order.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {
namespace WebRTC {

static inline bool isStunMessage(uint16_t messageType)
{
    // https://tools.ietf.org/html/rfc5389#section-6 for STUN messages.
    // TURN messages start by the channel number which is constrained by https://tools.ietf.org/html/rfc5766#section-11.
    return !(messageType & 0xC000);
}

std::optional<STUNMessageLengths> getSTUNOrTURNMessageLengths(std::span<const uint8_t> data)
{
    if (data.size() < 4)
        return { };

    auto messageType = be16toh(reinterpretCastSpanStartTo<const uint16_t>(data));
    auto messageLength = be16toh(reinterpretCastSpanStartTo<const uint16_t>(data.subspan(2)));

    // STUN data message header is 20 bytes.
    if (isStunMessage(messageType)) {
        size_t length = 20 + messageLength;
        return STUNMessageLengths { length, length };
    }

    // TURN data message header is 4 bytes plus padding bytes to get 4 bytes alignment as needed.
    size_t length = 4 + messageLength;
    size_t roundedLength = length % 4 ? (length + 4 - (length % 4)) : length;
    return STUNMessageLengths { length, roundedLength };
}

static inline Vector<uint8_t> extractSTUNOrTURNMessages(Vector<uint8_t>&& buffered, const Function<void(std::span<const uint8_t> data)>& processMessage)
{
    auto data = buffered.span();

    while (true) {
        auto lengths = getSTUNOrTURNMessageLengths(data);

        if (!lengths || lengths->messageLengthWithPadding > data.size()) {
            if (!data.size())
                return { };

            memcpySpan(buffered.mutableSpan(), data);
            buffered.resize(data.size());
            return WTFMove(buffered);
        }

        processMessage(data.first(lengths->messageLength));

        skip(data, lengths->messageLengthWithPadding);
    }
}

static inline Vector<uint8_t> extractDataMessages(Vector<uint8_t>&& buffered, const Function<void(std::span<const uint8_t> data)>& processMessage)
{
    constexpr size_t lengthFieldSize = sizeof(uint16_t); // number of bytes read by be16toh.

    auto data = buffered.span();

    while (true) {
        bool canReadLength = data.size() >= lengthFieldSize;
        size_t length = canReadLength ? be16toh(reinterpretCastSpanStartTo<const uint16_t>(data)) : 0;
        if (!canReadLength || length > data.size() - lengthFieldSize) {
            if (data.empty())
                return { };

            memcpySpan(buffered.mutableSpan(), data);
            buffered.shrink(data.size());
            return WTFMove(buffered);
        }

        skip(data, lengthFieldSize);

        processMessage(consumeSpan(data, length));
    }
}

Vector<uint8_t> extractMessages(Vector<uint8_t>&& buffer, MessageType type, const Function<void(std::span<const uint8_t> data)>& processMessage)
{
    return type == MessageType::STUN ? extractSTUNOrTURNMessages(WTFMove(buffer), processMessage) : extractDataMessages(WTFMove(buffer), processMessage);
}

} // namespace WebRTC
} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
