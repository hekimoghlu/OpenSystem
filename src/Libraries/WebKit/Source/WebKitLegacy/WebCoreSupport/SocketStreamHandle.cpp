/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#include "SocketStreamHandle.h"

#include "SocketStreamHandleClient.h"
#include <WebCore/CookieRequestHeaderFieldProxy.h>
#include <wtf/Function.h>

namespace WebCore {

SocketStreamHandle::SocketStreamHandle(const URL& url, SocketStreamHandleClient& client)
    : m_url(url)
    , m_client(client)
    , m_state(Connecting)
{
    ASSERT(isMainThread());
}

SocketStreamHandle::SocketStreamState SocketStreamHandle::state() const
{
    return m_state;
}

void SocketStreamHandle::sendData(std::span<const uint8_t> data, Function<void(bool)> completionHandler)
{
    if (m_state == Connecting || m_state == Closing)
        return completionHandler(false);
    platformSend(data, WTFMove(completionHandler));
}

void SocketStreamHandle::sendHandshake(CString&& handshake, std::optional<CookieRequestHeaderFieldProxy>&& headerFieldProxy, Function<void(bool, bool)> completionHandler)
{
    if (m_state == Connecting || m_state == Closing)
        return completionHandler(false, false);
    platformSendHandshake(handshake.span(), WTFMove(headerFieldProxy), WTFMove(completionHandler));
}

void SocketStreamHandle::close()
{
    if (m_state == Closed)
        return;
    m_state = Closing;
    if (bufferedAmount())
        return;
    disconnect();
}

void SocketStreamHandle::disconnect()
{
    Ref protect = static_cast<SocketStreamHandle&>(*this); // platformClose calls the client, which may make the handle get deallocated immediately.

    platformClose();
    m_state = Closed;
}

} // namespace WebCore
