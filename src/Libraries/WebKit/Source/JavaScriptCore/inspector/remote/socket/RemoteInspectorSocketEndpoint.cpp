/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include "RemoteInspectorSocketEndpoint.h"

#if ENABLE(REMOTE_INSPECTOR)

#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteInspectorSocketEndpoint);
WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteInspectorSocketEndpoint::BaseConnection);
WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteInspectorSocketEndpoint::ClientConnection);

RemoteInspectorSocketEndpoint& RemoteInspectorSocketEndpoint::singleton()
{
    static LazyNeverDestroyed<RemoteInspectorSocketEndpoint> shared;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        shared.construct();
    });
    return shared;
}

RemoteInspectorSocketEndpoint::RemoteInspectorSocketEndpoint()
{
    Socket::init();

    if (auto sockets = Socket::createPair()) {
        m_wakeupSendSocket = sockets->at(0);
        m_wakeupReceiveSocket = sockets->at(1);
    }

    m_workerThread = Thread::create("SocketEndpoint"_s, [this] {
        workerThread();
    });
}

RemoteInspectorSocketEndpoint::~RemoteInspectorSocketEndpoint()
{
    ASSERT(m_workerThread.get() != &Thread::current());

    m_shouldAbortWorkerThread = true;
    wakeupWorkerThread();
    m_workerThread->waitForCompletion();

    Socket::close(m_wakeupSendSocket);
    Socket::close(m_wakeupReceiveSocket);
    for (const auto& connection : m_clients.values())
        Socket::close(connection->socket);
    for (const auto& connection : m_listeners.values())
        Socket::close(connection->socket);
}

void RemoteInspectorSocketEndpoint::wakeupWorkerThread()
{
    if (Socket::isValid(m_wakeupSendSocket))
        Socket::write(m_wakeupSendSocket, "1", 1);
}

std::optional<ConnectionID> RemoteInspectorSocketEndpoint::connectInet(const char* serverAddress, uint16_t serverPort, Client& client)
{
    if (auto socket = Socket::connect(serverAddress, serverPort))
        return createClient(*socket, client);
    return std::nullopt;
}

std::optional<ConnectionID> RemoteInspectorSocketEndpoint::listenInet(const char* address, uint16_t port, Listener& listener)
{
    Locker locker { m_connectionsLock };
    auto id = generateConnectionID();
    auto connection = makeUnique<ListenerConnection>(id, listener, address, port);
    if (!connection->isListening())
        return std::nullopt;

    m_listeners.add(id, WTFMove(connection));
    wakeupWorkerThread();
    return id;
}

bool RemoteInspectorSocketEndpoint::isListening(ConnectionID id)
{
    Locker locker { m_connectionsLock };
    if (m_listeners.contains(id))
        return true;
    return false;
}

int RemoteInspectorSocketEndpoint::pollingTimeout()
{
    std::optional<MonotonicTime> mostRecentWakeup;
    for (const auto& connection : m_listeners) {
        if (connection.value->nextRetryTime) {
            if (mostRecentWakeup)
                mostRecentWakeup = std::min<MonotonicTime>(*mostRecentWakeup, *connection.value->nextRetryTime);
            else
                mostRecentWakeup = connection.value->nextRetryTime;
        }
    }

    if (mostRecentWakeup)
        return static_cast<int>((*mostRecentWakeup - MonotonicTime::now()).milliseconds());

    return -1;
}

void RemoteInspectorSocketEndpoint::workerThread()
{
    PollingDescriptor wakeup = Socket::preparePolling(m_wakeupReceiveSocket);

#if USE(GENERIC_EVENT_LOOP) || USE(WINDOWS_EVENT_LOOP)
    RunLoop::setWakeUpCallback([this] {
        wakeupWorkerThread();
    });
#endif

    while (!m_shouldAbortWorkerThread) {
#if USE(GENERIC_EVENT_LOOP) || USE(WINDOWS_EVENT_LOOP)
        RunLoop::cycle();
#endif

        Vector<PollingDescriptor> pollfds;
        Vector<ConnectionID> ids;
        {
            Locker locker { m_connectionsLock };
            for (const auto& connection : m_clients) {
                pollfds.append(connection.value->poll);
                ids.append(connection.key);
            }
            for (const auto& connection : m_listeners) {
                if (!connection.value->isListening() && connection.value->listen())
                    connection.value->listener.didChangeStatus(*this, connection.key, Listener::Status::Listening);
                if (connection.value->isListening()) {
                    pollfds.append(connection.value->poll);
                    ids.append(connection.key);
                }
            }
        }
        pollfds.append(wakeup);

        if (!Socket::poll(pollfds, pollingTimeout()))
            continue;

        if (Socket::isReadable(pollfds.last())) {
            char wakeMessage;
            Socket::read(m_wakeupReceiveSocket, &wakeMessage, sizeof(wakeMessage));
            continue;
        }

        for (size_t i = 0; i < ids.size(); i++) {
            auto id = ids[i];

            if (Socket::isReadable(pollfds[i])) {
                if (isListening(id))
                    acceptInetSocketIfEnabled(id);
                else
                    recvIfEnabled(id);
            } else if (Socket::isWritable(pollfds[i]))
                sendIfEnabled(id);
        }
    }

#if USE(GENERIC_EVENT_LOOP) || USE(WINDOWS_EVENT_LOOP)
    RunLoop::setWakeUpCallback(WTF::Function<void()>());
#endif
}

ConnectionID RemoteInspectorSocketEndpoint::generateConnectionID()
{
    ASSERT(m_connectionsLock.isLocked());
    ConnectionID id;
    do {
        id = cryptographicallyRandomNumber<ConnectionID>();
    } while (!m_clients.isValidKey(id) || m_clients.contains(id) || m_listeners.contains(id));
    return id;
}

std::optional<ConnectionID> RemoteInspectorSocketEndpoint::createClient(PlatformSocketType socket, Client& client)
{
    ASSERT(Socket::isValid(socket));

    Locker locker { m_connectionsLock };
    auto id = generateConnectionID();
    auto connection = makeUnique<ClientConnection>(id, socket, client);
    if (!Socket::isValid(connection->socket))
        return std::nullopt;

    m_clients.add(id, WTFMove(connection));
    wakeupWorkerThread();

    return id;
}

void RemoteInspectorSocketEndpoint::disconnect(ConnectionID id)
{
    Locker locker { m_connectionsLock };

    if (const auto& connection = m_listeners.get(id)) {
        m_listeners.remove(id);
        Socket::close(connection->socket);
        locker.unlockEarly();
        connection->listener.didChangeStatus(*this, id, Listener::Status::Closed);
    } else if (const auto& connection = m_clients.get(id)) {
        m_clients.remove(id);
        Socket::close(connection->socket);
        locker.unlockEarly();
        connection->client.didClose(*this, id);
    } else
        LOG_ERROR("Error: Cannot disconnect: Invalid id");
}

void RemoteInspectorSocketEndpoint::invalidateClient(Client& client)
{
    Locker locker { m_connectionsLock };
    m_clients.removeIf([&client](auto& keyValue) {
        const auto& connection = keyValue.value;

        if (&connection->client != &client)
            return false;

        Socket::close(connection->socket);
        // do not call client.didClose because client is already invalidating phase.
        return true;
    });
}

void RemoteInspectorSocketEndpoint::invalidateListener(Listener& listener)
{
    Locker locker { m_connectionsLock };
    m_listeners.removeIf([&listener](auto& keyValue) {
        const auto& connection = keyValue.value;

        if (&connection->listener == &listener) {
            Socket::close(connection->socket);
            return true;
        }

        return false;
    });
}

std::optional<uint16_t> RemoteInspectorSocketEndpoint::getPort(ConnectionID id) const
{
    Locker locker { m_connectionsLock };
    if (const auto& connection = m_listeners.get(id))
        return Socket::getPort(connection->socket);
    if (const auto& connection = m_clients.get(id))
        return Socket::getPort(connection->socket);

    return std::nullopt;
}

void RemoteInspectorSocketEndpoint::recvIfEnabled(ConnectionID id)
{
    Locker locker { m_connectionsLock };
    if (const auto& connection = m_clients.get(id)) {
        Vector<uint8_t> recvBuffer(Socket::BufferSize);
        if (auto readSize = Socket::read(connection->socket, recvBuffer.data(), recvBuffer.size())) {
            if (*readSize > 0) {
                recvBuffer.shrink(*readSize);
                locker.unlockEarly();
                connection->client.didReceive(*this, id, WTFMove(recvBuffer));
                return;
            }
        }

        Socket::close(connection->socket);
        m_clients.remove(id);

        locker.unlockEarly();
        connection->client.didClose(*this, id);
    }
}

void RemoteInspectorSocketEndpoint::sendIfEnabled(ConnectionID id)
{
    Locker locker { m_connectionsLock };
    if (const auto& connection = m_clients.get(id)) {
        Socket::clearWaitingWritable(connection->poll);

        auto& buffer = connection->sendBuffer;
        if (buffer.isEmpty())
            return;

        if (auto writeSize = Socket::write(connection->socket, buffer.data(), std::min(buffer.size(), Socket::BufferSize))) {
            auto size = *writeSize;
            if (size == buffer.size()) {
                buffer.clear();
                return;
            }

            if (size > 0)
                buffer.remove(0, size);
        }

        Socket::markWaitingWritable(connection->poll);
    }
}

void RemoteInspectorSocketEndpoint::send(ConnectionID id, std::span<const uint8_t> data)
{
    Locker locker { m_connectionsLock };
    if (const auto& connection = m_clients.get(id)) {
        size_t offset = 0;
        if (connection->sendBuffer.isEmpty()) {
            // Try to call send() directly if buffer is empty.
            if (auto writeSize = Socket::write(connection->socket, data.data(), std::min(data.size(), Socket::BufferSize)))
                offset = *writeSize;
            // @TODO need to handle closed socket case?
        }

        // Check all data is sent.
        if (offset == data.size())
            return;

        // Copy remaining data to send later.
        connection->sendBuffer.append(data.subspan(offset));
        Socket::markWaitingWritable(connection->poll);

        wakeupWorkerThread();
    }
}

void RemoteInspectorSocketEndpoint::acceptInetSocketIfEnabled(ConnectionID id)
{
    ASSERT(isListening(id));

    Locker locker { m_connectionsLock };
    if (const auto& connection = m_listeners.get(id)) {
        if (auto socket = Socket::accept(connection->socket)) {
            // Need to unlock before calling createClient as it also attempts to lock.
            locker.unlockEarly();
            if (connection->listener.doAccept(*this, socket.value()))
                return;

            Socket::close(*socket);
        } else {
            // If accept() returns error, we have to start over with bind() and listen().
            // By closing socket here, listen() will be called again at the next loop of worker thread.
            Socket::close(connection->socket);
            connection->listener.didChangeStatus(*this, id, Listener::Status::Invalid);
        }
    }
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
