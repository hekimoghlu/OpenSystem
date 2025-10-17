/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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

#include <span>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(COCOA)
#include <Network/Network.h>
#include <wtf/RetainPtr.h>
#endif

namespace WebCore {
class Exception;
struct WebTransportStreamIdentifierType;
using WebTransportStreamIdentifier = ObjectIdentifier<WebTransportStreamIdentifierType>;
using WebTransportStreamErrorCode = uint64_t;
}

namespace WebKit {
enum class NetworkTransportStreamType : uint8_t { Bidirectional, OutgoingUnidirectional, IncomingUnidirectional };
enum class NetworkTransportStreamState : uint8_t { Ready, ReadClosed, WriteClosed };

class NetworkTransportSession;

class NetworkTransportStream : public RefCounted<NetworkTransportStream>, public CanMakeWeakPtr<NetworkTransportStream> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkTransportStream);
public:
    template<typename... Args> static Ref<NetworkTransportStream> create(Args&&... args) { return adoptRef(*new NetworkTransportStream(std::forward<Args>(args)...)); }

    WebCore::WebTransportStreamIdentifier identifier() const { return m_identifier; }

    void sendBytes(std::span<const uint8_t>, bool, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&);
    void cancelReceive(std::optional<WebCore::WebTransportStreamErrorCode>);
    void cancelSend(std::optional<WebCore::WebTransportStreamErrorCode>);
    void cancel(std::optional<WebCore::WebTransportStreamErrorCode>);
protected:
#if PLATFORM(COCOA)
    NetworkTransportStream(NetworkTransportSession&, nw_connection_t, NetworkTransportStreamType);
#else
    NetworkTransportStream();
#endif

private:
    void receiveLoop();
    void setErrorCodeForStream(std::optional<WebCore::WebTransportStreamErrorCode>);

    const WebCore::WebTransportStreamIdentifier m_identifier;
    WeakPtr<NetworkTransportSession> m_session;
#if PLATFORM(COCOA)
    const RetainPtr<nw_connection_t> m_connection;
#endif
    const NetworkTransportStreamType m_streamType;
    NetworkTransportStreamState m_streamState;
};

}
