/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspectorSocket.h"
#include <wtf/Condition.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class RemoteInspectorSocketEndpoint {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(RemoteInspectorSocketEndpoint, JS_EXPORT_PRIVATE);
public:
    class Client {
    public:
        virtual ~Client() { }

        // These callbacks are not guaranteed to be called from the main thread.
        virtual void didReceive(RemoteInspectorSocketEndpoint&, ConnectionID, Vector<uint8_t>&&) = 0;
        virtual void didClose(RemoteInspectorSocketEndpoint&, ConnectionID) = 0;
    };

    class Listener {
    public:
        enum class Status : uint8_t {
            Listening,
            Invalid,
            Closed,
        };
        virtual ~Listener() { }

        // These callbacks are not guaranteed to be called from the main thread.
        virtual std::optional<ConnectionID> doAccept(RemoteInspectorSocketEndpoint&, PlatformSocketType) = 0;
        virtual void didChangeStatus(RemoteInspectorSocketEndpoint&, ConnectionID, Status) = 0;
    };

    JS_EXPORT_PRIVATE static RemoteInspectorSocketEndpoint& singleton();

    RemoteInspectorSocketEndpoint();
    JS_EXPORT_PRIVATE ~RemoteInspectorSocketEndpoint();

    std::optional<ConnectionID> connectInet(const char* serverAddr, uint16_t serverPort, Client&);
    JS_EXPORT_PRIVATE std::optional<ConnectionID> listenInet(const char* address, uint16_t port, Listener&);
    void invalidateClient(Client&);
    void invalidateListener(Listener&);

    JS_EXPORT_PRIVATE void send(ConnectionID, std::span<const uint8_t>);

    JS_EXPORT_PRIVATE std::optional<ConnectionID> createClient(PlatformSocketType, Client&);

    std::optional<uint16_t> getPort(ConnectionID) const;

    JS_EXPORT_PRIVATE void disconnect(ConnectionID);

protected:
    struct BaseConnection {
        WTF_MAKE_STRUCT_TZONE_ALLOCATED(BaseConnection);

        BaseConnection(ConnectionID id)
            : id { id }
            , socket { INVALID_SOCKET_VALUE }
        {
        }

        bool setSocket(PlatformSocketType newSocket)
        {
            ASSERT(Socket::isValid(newSocket));

            if (!Socket::setup(newSocket))
                return false;

            if (Socket::isValid(socket))
                Socket::close(socket);

            socket = newSocket;
            poll = Socket::preparePolling(socket);
            return true;
        }

        ConnectionID id;
        PlatformSocketType socket;
        PollingDescriptor poll;
    };

    struct ClientConnection : public BaseConnection {
        WTF_MAKE_STRUCT_TZONE_ALLOCATED(ClientConnection);
        ClientConnection(ConnectionID id, PlatformSocketType socket, Client& client)
            : BaseConnection(id)
            , client { client }
        {
            setSocket(socket);
        }

        Client& client;
        Vector<uint8_t> sendBuffer;
    };

    struct ListenerConnection : public BaseConnection {
        static constexpr Seconds initialRetryInterval { 200_ms };
        static constexpr Seconds maxRetryInterval { 5_s };

        ListenerConnection(ConnectionID id, Listener& listener, const char* address, uint16_t port)
            : BaseConnection(id)
            , address { String::fromLatin1(address) }
            , port { port }
            , listener { listener }
        {
            listen();
        }

        bool listen()
        {
            ASSERT(!isListening());

            if (nextRetryTime && *nextRetryTime > MonotonicTime::now())
                return false;

            if (auto newSocket = Socket::listen(address.utf8().data(), port)) {
                if (setSocket(*newSocket)) {
                    retryInterval = initialRetryInterval;
                    return true;
                }
                Socket::close(*newSocket);
            }

            nextRetryTime = MonotonicTime::now() + retryInterval;
            retryInterval = std::min<Seconds>(retryInterval * 2, maxRetryInterval);

            return false;
        }

        bool isListening()
        {
            return Socket::isListening(socket);
        }

        String address;
        uint16_t port;
        Listener& listener;
        std::optional<MonotonicTime> nextRetryTime;
        Seconds retryInterval { initialRetryInterval };
    };

    ConnectionID generateConnectionID();

    void recvIfEnabled(ConnectionID);
    void sendIfEnabled(ConnectionID);
    void workerThread();
    void wakeupWorkerThread();
    void acceptInetSocketIfEnabled(ConnectionID);
    bool isListening(ConnectionID);
    int pollingTimeout();

    mutable Lock m_connectionsLock;
    UncheckedKeyHashMap<ConnectionID, std::unique_ptr<ClientConnection>> m_clients WTF_GUARDED_BY_LOCK(m_connectionsLock);
    UncheckedKeyHashMap<ConnectionID, std::unique_ptr<ListenerConnection>> m_listeners WTF_GUARDED_BY_LOCK(m_connectionsLock);

    PlatformSocketType m_wakeupSendSocket { INVALID_SOCKET_VALUE };
    PlatformSocketType m_wakeupReceiveSocket { INVALID_SOCKET_VALUE };

    RefPtr<Thread> m_workerThread;
    std::atomic<bool> m_shouldAbortWorkerThread { false };
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
