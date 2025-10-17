/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/URL.h>

namespace WebCore {

struct CookieRequestHeaderFieldProxy;
class SocketStreamHandleClient;

struct SourceApplicationAuditToken {
#if PLATFORM(COCOA)
    RetainPtr<CFDataRef> sourceApplicationAuditData;
#else
    void *empty { nullptr };
#endif
};

class SocketStreamHandle : public ThreadSafeRefCounted<SocketStreamHandle, WTF::DestructionThread::Main> {
public:
    enum SocketStreamState { Connecting, Open, Closing, Closed };
    virtual ~SocketStreamHandle() = default;
    SocketStreamState state() const;

    void sendData(std::span<const uint8_t> data, Function<void(bool)>);
    void sendHandshake(CString&& handshake, std::optional<CookieRequestHeaderFieldProxy>&&, Function<void(bool, bool)>);
    void close(); // Disconnect after all data in buffer are sent.
    void disconnect();
    virtual size_t bufferedAmount() = 0;

protected:
    WEBCORE_EXPORT SocketStreamHandle(const URL&, SocketStreamHandleClient&);

    virtual void platformSend(std::span<const uint8_t> data, Function<void(bool)>&&) = 0;
    virtual void platformSendHandshake(std::span<const uint8_t> data, const std::optional<CookieRequestHeaderFieldProxy>&, Function<void(bool, bool)>&&) = 0;
    virtual void platformClose() = 0;

    URL m_url;
    SocketStreamHandleClient& m_client;
    SocketStreamState m_state;
};

} // namespace WebCore
