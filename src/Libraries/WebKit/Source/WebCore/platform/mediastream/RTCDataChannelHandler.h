/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

#if ENABLE(WEB_RTC)

#include "RTCPriorityType.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

struct RTCDataChannelInit {
    std::optional<bool> ordered;
    std::optional<unsigned short> maxPacketLifeTime;
    std::optional<unsigned short> maxRetransmits;
    String protocol;
    std::optional<bool> negotiated;
    std::optional<unsigned short> id;
    RTCPriorityType priority { RTCPriorityType::Low };

    RTCDataChannelInit isolatedCopy() const &;
    RTCDataChannelInit isolatedCopy() &&;
};

inline RTCDataChannelInit RTCDataChannelInit::isolatedCopy() const &
{
    auto copy = *this;
    copy.protocol = protocol.isolatedCopy();
    return copy;
}

inline RTCDataChannelInit RTCDataChannelInit::isolatedCopy() &&
{
    auto copy = WTFMove(*this);
    copy.protocol = WTFMove(copy.protocol).isolatedCopy();
    return copy;
}

class RTCDataChannelHandlerClient;

class RTCDataChannelHandler {
public:
    virtual ~RTCDataChannelHandler() = default;

    virtual void setClient(RTCDataChannelHandlerClient&, std::optional<ScriptExecutionContextIdentifier>) = 0;

    virtual bool sendStringData(const CString&) = 0;
    virtual bool sendRawData(std::span<const uint8_t>) = 0;
    virtual void close() = 0;

    virtual std::optional<unsigned short> id() const { return { }; }
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
