/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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

#include "ActiveDOMObject.h"
#include "ExceptionOr.h"
#include "WebTransportOptions.h"
#include "WebTransportReliabilityMode.h"
#include "WebTransportSessionClient.h"
#include <wtf/ListHashSet.h>
#include <wtf/RefCounted.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

enum class WebTransportCongestionControl : uint8_t;

class DOMException;
class DOMPromise;
class DatagramSource;
class DeferredPromise;
class Exception;
class JSDOMGlobalObject;
class ReadableStream;
class ReadableStreamSource;
class ScriptExecutionContext;
class SocketProvider;
class WebTransportBidirectionalStreamSource;
class WebTransportDatagramDuplexStream;
class WebTransportError;
class WebTransportReceiveStreamSource;
class WebTransportSession;
class WorkerWebTransportSession;
class WritableStream;

struct WebTransportBidirectionalStreamConstructionParameters;
struct WebTransportCloseInfo;
struct WebTransportSendStreamOptions;
struct WebTransportHash;

class WebTransport : public WebTransportSessionClient, public ActiveDOMObject {
public:
    static ExceptionOr<Ref<WebTransport>> create(ScriptExecutionContext&, String&&, WebTransportOptions&&);
    ~WebTransport();

    // ActiveDOMObject.
    void ref() const final { WebTransportSessionClient::ref(); }
    void deref() const final { WebTransportSessionClient::deref(); }

    void getStats(Ref<DeferredPromise>&&);
    DOMPromise& ready();
    WebTransportReliabilityMode reliability();
    WebTransportCongestionControl congestionControl();
    DOMPromise& closed();
    DOMPromise& draining();
    void close(WebTransportCloseInfo&&);
    WebTransportDatagramDuplexStream& datagrams();
    void createBidirectionalStream(ScriptExecutionContext&, WebTransportSendStreamOptions&&, Ref<DeferredPromise>&&);
    ReadableStream& incomingBidirectionalStreams();
    void createUnidirectionalStream(ScriptExecutionContext&, WebTransportSendStreamOptions&&, Ref<DeferredPromise>&&);
    ReadableStream& incomingUnidirectionalStreams();

    RefPtr<WebTransportSession> session();

private:
    WebTransport(ScriptExecutionContext&, JSDOMGlobalObject&, Ref<ReadableStream>&&, Ref<ReadableStream>&&, WebTransportCongestionControl, Ref<WebTransportDatagramDuplexStream>&&, Ref<DatagramSource>&&, Ref<WebTransportReceiveStreamSource>&&, Ref<WebTransportBidirectionalStreamSource>&&);

    void initializeOverHTTP(SocketProvider&, ScriptExecutionContext&, URL&&, bool dedicated, bool http3Only, WebTransportCongestionControl, Vector<WebTransportHash>&&);
    void cleanup(Ref<DOMException>&&, std::optional<WebTransportCloseInfo>&&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    void receiveDatagram(std::span<const uint8_t>, bool, std::optional<Exception>&&) final;
    void receiveIncomingUnidirectionalStream(WebTransportStreamIdentifier) final;
    void receiveBidirectionalStream(WebTransportBidirectionalStreamConstructionParameters&&) final;
    void streamReceiveBytes(WebTransportStreamIdentifier, std::span<const uint8_t>, bool, std::optional<Exception>&&) final;
    void networkProcessCrashed() final;

    ListHashSet<Ref<WritableStream>> m_sendStreams;
    ListHashSet<Ref<ReadableStream>> m_receiveStreams;
    Ref<ReadableStream> m_incomingBidirectionalStreams;
    Ref<ReadableStream> m_incomingUnidirectionalStreams;

    // https://www.w3.org/TR/webtransport/#dom-webtransport-state-slot
    enum class State : uint8_t {
        Connecting,
        Connected,
        Draining,
        Closed,
        Failed
    };
    State m_state { State::Connecting };

    using PromiseAndWrapper = std::pair<Ref<DOMPromise>, Ref<DeferredPromise>>;
    PromiseAndWrapper m_ready;
    WebTransportReliabilityMode m_reliability { WebTransportReliabilityMode::Pending };
    WebTransportCongestionControl m_congestionControl;
    PromiseAndWrapper m_closed;
    PromiseAndWrapper m_draining;
    Ref<WebTransportDatagramDuplexStream> m_datagrams;
    RefPtr<WebTransportSession> m_session;
    Ref<DatagramSource> m_datagramSource;
    Ref<WebTransportReceiveStreamSource> m_receiveStreamSource;
    Ref<WebTransportBidirectionalStreamSource> m_bidirectionalStreamSource;
    HashMap<WebTransportStreamIdentifier, Ref<WebTransportReceiveStreamSource>> m_readStreamSources;
};

}
