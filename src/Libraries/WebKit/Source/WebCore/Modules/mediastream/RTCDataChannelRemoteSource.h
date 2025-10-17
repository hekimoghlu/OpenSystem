/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

#include "RTCDataChannelHandler.h"
#include "RTCDataChannelHandlerClient.h"
#include "RTCDataChannelIdentifier.h"
#include "RTCDataChannelRemoteSourceConnection.h"
#include "RTCError.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class RTCDataChannelRemoteSource : public RTCDataChannelHandlerClient {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(RTCDataChannelRemoteSource, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT RTCDataChannelRemoteSource(RTCDataChannelIdentifier, UniqueRef<RTCDataChannelHandler>&&, Ref<RTCDataChannelRemoteSourceConnection>&&);
    ~RTCDataChannelRemoteSource();

    void sendStringData(const CString& text) { m_handler->sendStringData(text); }
    void sendRawData(std::span<const uint8_t> data) { m_handler->sendRawData(data); }
    void close() { m_handler->close(); }

private:

    // RTCDataChannelHandlerClient
    void didChangeReadyState(RTCDataChannelState state) final { m_connection->didChangeReadyState(m_identifier, state); }
    void didReceiveStringData(const String& text) final { m_connection->didReceiveStringData(m_identifier, text); }
    void didReceiveRawData(std::span<const uint8_t> data) final { m_connection->didReceiveRawData(m_identifier, data); }
    void didDetectError(Ref<RTCError>&& error) final { m_connection->didDetectError(m_identifier, error->errorDetail(), error->message()); }
    void bufferedAmountIsDecreasing(size_t amount) final { m_connection->bufferedAmountIsDecreasing(m_identifier, amount); }
    size_t bufferedAmount() const final { return 0; }

    RTCDataChannelIdentifier m_identifier;
    UniqueRef<RTCDataChannelHandler> m_handler;
    Ref<RTCDataChannelRemoteSourceConnection> m_connection;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
