/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

#include "WorkQueueMessageReceiver.h"
#include <WebCore/ProcessQualified.h>
#include <WebCore/RTCDataChannelRemoteHandler.h>
#include <WebCore/RTCDataChannelRemoteHandlerConnection.h>
#include <WebCore/RTCDataChannelRemoteSource.h>
#include <WebCore/RTCDataChannelRemoteSourceConnection.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class RTCDataChannelRemoteManager final : private IPC::MessageReceiver {
public:
    static RTCDataChannelRemoteManager& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    WebCore::RTCDataChannelRemoteHandlerConnection& remoteHandlerConnection();
    bool connectToRemoteSource(WebCore::RTCDataChannelIdentifier source, WebCore::RTCDataChannelIdentifier handler);

private:
    RTCDataChannelRemoteManager();
    void initialize();

    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void sendData(WebCore::RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>);
    void close(WebCore::RTCDataChannelIdentifier);

    // To handler
    void changeReadyState(WebCore::RTCDataChannelIdentifier, WebCore::RTCDataChannelState);
    void receiveData(WebCore::RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>);
    void detectError(WebCore::RTCDataChannelIdentifier, WebCore::RTCErrorDetailType, String&&);
    void bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier, size_t);

    WebCore::RTCDataChannelRemoteSourceConnection& remoteSourceConnection();
    void postTaskToHandler(WebCore::RTCDataChannelIdentifier, Function<void(WebCore::RTCDataChannelRemoteHandler&)>&&);
    WebCore::RTCDataChannelRemoteSource* sourceFromIdentifier(WebCore::RTCDataChannelIdentifier);

    class RemoteHandlerConnection : public WebCore::RTCDataChannelRemoteHandlerConnection {
    public:
        static Ref<RemoteHandlerConnection> create(Ref<WorkQueue>&&);

        void connectToSource(WebCore::RTCDataChannelRemoteHandler&, std::optional<WebCore::ScriptExecutionContextIdentifier>, WebCore::RTCDataChannelIdentifier, WebCore::RTCDataChannelIdentifier) final;
        void sendData(WebCore::RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>) final;
        void close(WebCore::RTCDataChannelIdentifier) final;

    private:
        explicit RemoteHandlerConnection(Ref<WorkQueue>&&);

        Ref<IPC::Connection> m_connection;
        Ref<WorkQueue> m_queue;
    };

    class RemoteSourceConnection : public WebCore::RTCDataChannelRemoteSourceConnection {
    public:
        static Ref<RemoteSourceConnection> create();

    private:
        RemoteSourceConnection();

        void didChangeReadyState(WebCore::RTCDataChannelIdentifier, WebCore::RTCDataChannelState) final;
        void didReceiveStringData(WebCore::RTCDataChannelIdentifier, const String&) final;
        void didReceiveRawData(WebCore::RTCDataChannelIdentifier, std::span<const uint8_t>) final;
        void didDetectError(WebCore::RTCDataChannelIdentifier, WebCore::RTCErrorDetailType, const String&) final;
        void bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier, size_t) final;

        Ref<IPC::Connection> m_connection;
    };

    struct RemoteHandler {
        WeakPtr<WebCore::RTCDataChannelRemoteHandler> handler;
        Markable<WebCore::ScriptExecutionContextIdentifier> contextIdentifier;
    };

    Ref<WorkQueue> m_queue;
    RefPtr<IPC::Connection> m_connection;
    RefPtr<RemoteHandlerConnection> m_remoteHandlerConnection;
    RefPtr<RemoteSourceConnection> m_remoteSourceConnection;
    HashMap<WebCore::RTCDataChannelLocalIdentifier, UniqueRef<WebCore::RTCDataChannelRemoteSource>> m_sources;
    HashMap<WebCore::RTCDataChannelLocalIdentifier, RemoteHandler> m_handlers;
};

} // namespace WebKit

#endif // ENABLE(WEB_RTC)
