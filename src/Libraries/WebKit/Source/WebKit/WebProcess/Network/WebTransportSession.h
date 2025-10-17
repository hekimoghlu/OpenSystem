/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/ProcessQualified.h>
#include <WebCore/WebTransportSession.h>
#include <wtf/NativePromise.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace IPC {
enum class Error : uint8_t;
}

namespace WebCore {
class Exception;
class WebTransportSession;
class WebTransportSessionClient;
struct ClientOrigin;
struct WebTransportStreamIdentifierType;
using WebTransportStreamIdentifier = ObjectIdentifier<WebTransportStreamIdentifierType>;
using WebTransportSessionPromise = NativePromise<Ref<WebTransportSession>, void>;
using WebTransportSendPromise = NativePromise<std::optional<Exception>, void>;
using WebTransportSessionErrorCode = uint32_t;
using WebTransportStreamErrorCode = uint64_t;
}

namespace WebKit {

class WebTransportBidirectionalStream;
class WebTransportReceiveStream;
class WebTransportSendStream;

struct WebTransportSessionIdentifierType { };

using WebTransportSessionIdentifier = ObjectIdentifier<WebTransportSessionIdentifierType>;

class WebTransportSession : public WebCore::WebTransportSession, public IPC::MessageReceiver, public IPC::MessageSender, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WebTransportSession, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<WebCore::WebTransportSessionPromise> initialize(Ref<IPC::Connection>&&, ThreadSafeWeakPtr<WebCore::WebTransportSessionClient>&&, const URL&, const WebPageProxyIdentifier&, const WebCore::ClientOrigin&);
    ~WebTransportSession();

    void receiveDatagram(std::span<const uint8_t>, bool, std::optional<WebCore::Exception>&&);
    void receiveIncomingUnidirectionalStream(WebCore::WebTransportStreamIdentifier);
    void receiveBidirectionalStream(WebCore::WebTransportStreamIdentifier);
    void streamReceiveBytes(WebCore::WebTransportStreamIdentifier, std::span<const uint8_t>, bool, std::optional<WebCore::Exception>&&);

    Ref<WebCore::WebTransportSendPromise> streamSendBytes(WebCore::WebTransportStreamIdentifier, std::span<const uint8_t>, bool withFin);

    void cancelReceiveStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);
    void cancelSendStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);
    void destroyStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);

    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    // MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    void networkProcessCrashed();
private:
    WebTransportSession(Ref<IPC::Connection>&&, ThreadSafeWeakPtr<WebCore::WebTransportSessionClient>&&, WebTransportSessionIdentifier);

    // WebTransportSession
    Ref<WebCore::WebTransportSendPromise> sendDatagram(std::span<const uint8_t>) final;
    Ref<WebCore::WritableStreamPromise> createOutgoingUnidirectionalStream() final;
    Ref<WebCore::BidirectionalStreamPromise> createBidirectionalStream() final;
    void terminate(WebCore::WebTransportSessionErrorCode, CString&&) final;

    // MessageSender
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    const Ref<IPC::Connection> m_connection;
    const ThreadSafeWeakPtr<WebCore::WebTransportSessionClient> m_client;
    const WebTransportSessionIdentifier m_identifier;
};

}
