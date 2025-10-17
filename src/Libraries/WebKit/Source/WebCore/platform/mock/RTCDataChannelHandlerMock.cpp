/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#include "RTCDataChannelHandlerMock.h"

#if ENABLE(WEB_RTC)

#include "ProcessQualified.h"
#include "RTCDataChannelHandlerClient.h"
#include "RTCNotifiersMock.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RTCDataChannelHandlerMock);

RTCDataChannelHandlerMock::RTCDataChannelHandlerMock(const String& label, const RTCDataChannelInit& init)
    : m_label(label)
    , m_protocol(init.protocol)
{
}

void RTCDataChannelHandlerMock::setClient(RTCDataChannelHandlerClient& client, std::optional<ScriptExecutionContextIdentifier>)
{
    ASSERT(!m_client);
    m_client = &client;
    auto notifier = adoptRef(*new DataChannelStateNotifier(m_client, RTCDataChannelState::Open));
    m_timerEvents.append(adoptRef(new TimerEvent(this, WTFMove(notifier))));
}

bool RTCDataChannelHandlerMock::sendStringData(const CString& string)
{
    m_client->didReceiveStringData(String::fromUTF8(string.span()));
    return true;
}

bool RTCDataChannelHandlerMock::sendRawData(std::span<const uint8_t> data)
{
    m_client->didReceiveRawData(data);
    return true;
}

void RTCDataChannelHandlerMock::close()
{
    auto notifier = adoptRef(*new DataChannelStateNotifier(m_client, RTCDataChannelState::Closed));
    m_timerEvents.append(adoptRef(new TimerEvent(this, WTFMove(notifier))));
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
