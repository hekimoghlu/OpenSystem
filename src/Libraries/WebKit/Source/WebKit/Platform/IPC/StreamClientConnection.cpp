/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "StreamClientConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace IPC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StreamClientConnection);

// FIXME(http://webkit.org/b/238986): Workaround for not being able to deliver messages from the dedicated connection to the work queue the client uses.

StreamClientConnection::DedicatedConnectionClient::DedicatedConnectionClient(StreamClientConnection& owner, Connection::Client& receiver)
    : m_owner(owner)
    , m_receiver(receiver)
{
}

void StreamClientConnection::DedicatedConnectionClient::didReceiveMessage(Connection& connection, Decoder& decoder)
{
    m_receiver.didReceiveMessage(connection, decoder);
}

bool StreamClientConnection::DedicatedConnectionClient::didReceiveSyncMessage(Connection& connection, Decoder& decoder, UniqueRef<Encoder>& replyEncoder)
{
    return m_receiver.didReceiveSyncMessage(connection, decoder, replyEncoder);
}

void StreamClientConnection::DedicatedConnectionClient::didClose(Connection& connection)
{
    // Client is expected to listen to Connection::didClose() from the connection it sent to the dedicated connection to.
    m_receiver.didClose(connection);
}

void StreamClientConnection::DedicatedConnectionClient::didReceiveInvalidMessage(Connection&, MessageName, int32_t)
{
    ASSERT_NOT_REACHED(); // The sender is expected to be trusted, so all invalid messages are programming errors.
}

std::optional<StreamClientConnection::StreamConnectionPair> StreamClientConnection::create(unsigned bufferSizeLog2, Seconds defaultTimeoutDuration)
{
    auto connectionIdentifiers = Connection::createConnectionIdentifierPair();
    if (!connectionIdentifiers)
        return std::nullopt;
    auto buffer = StreamClientConnectionBuffer::create(bufferSizeLog2);
    if (!buffer)
        return std::nullopt;
    // Create StreamClientConnection with "server" type Connection. The caller will send the "client" type connection identifier via
    // IPC to the other side, where StreamServerConnection will be created with "client" type Connection.
    // For Connection, "server" means the connection which was created first, the connection which is not sent through IPC to other party.
    // For Connection, "client" means the connection which was established by receiving it through IPC and creating IPC::Connection out from the identifier.
    // The "Client" in StreamClientConnection means the party that mostly does sending, e.g. untrusted party.
    // The "Server" in StreamServerConnection means the party that mostly does receiving, e.g. the trusted party which holds the destination object to communicate with.
    auto dedicatedConnection = Connection::createServerConnection(WTFMove(connectionIdentifiers->server));
    auto clientConnection = adoptRef(*new StreamClientConnection(WTFMove(dedicatedConnection), WTFMove(*buffer), defaultTimeoutDuration));
    StreamServerConnection::Handle serverHandle {
        WTFMove(connectionIdentifiers->client),
        clientConnection->m_buffer.createHandle()
    };
    return StreamClientConnection::StreamConnectionPair { WTFMove(clientConnection), WTFMove(serverHandle) };
}

StreamClientConnection::StreamClientConnection(Ref<Connection> connection, StreamClientConnectionBuffer&& buffer, Seconds defaultTimeoutDuration)
    : m_connection(WTFMove(connection))
    , m_buffer(WTFMove(buffer))
    , m_defaultTimeoutDuration(defaultTimeoutDuration)
{
}

StreamClientConnection::~StreamClientConnection()
{
    ASSERT(!m_connection->isValid());
}

void StreamClientConnection::setSemaphores(IPC::Semaphore&& wakeUp, IPC::Semaphore&& clientWait)
{
    m_buffer.setSemaphores(WTFMove(wakeUp), WTFMove(clientWait));
}

bool StreamClientConnection::hasSemaphores() const
{
    return m_buffer.hasSemaphores();
}

void StreamClientConnection::setMaxBatchSize(unsigned size)
{
    m_maxBatchSize = size;
    m_buffer.wakeUpServer();
}

void StreamClientConnection::open(Connection::Client& receiver, SerialFunctionDispatcher& dispatcher)
{
    m_dedicatedConnectionClient.emplace(*this, receiver);
    protectedConnection()->open(*m_dedicatedConnectionClient, dispatcher);
}

Error StreamClientConnection::flushSentMessages()
{
    auto timeout = defaultTimeout();
    wakeUpServer(WakeUpServer::Yes);
    return protectedConnection()->flushSentMessages(WTFMove(timeout));
}

void StreamClientConnection::invalidate()
{
    protectedConnection()->invalidate();
}

void StreamClientConnection::wakeUpServer(WakeUpServer wakeUpResult)
{
    if (wakeUpResult == WakeUpServer::No && !m_batchSize)
        return;
    m_buffer.wakeUpServer();
    m_batchSize = 0;
}

void StreamClientConnection::wakeUpServerBatched(WakeUpServer wakeUpResult)
{
    if (wakeUpResult == WakeUpServer::Yes || m_batchSize) {
        m_batchSize++;
        if (m_batchSize >= m_maxBatchSize)
            wakeUpServer(WakeUpServer::Yes);
    }
}

StreamClientConnectionBuffer& StreamClientConnection::bufferForTesting()
{
    return m_buffer;
}

Connection& StreamClientConnection::connectionForTesting()
{
    return m_connection.get();
}

void StreamClientConnection::addWorkQueueMessageReceiver(ReceiverName name, WorkQueue& workQueue, WorkQueueMessageReceiver& receiver, uint64_t destinationID)
{
    protectedConnection()->addWorkQueueMessageReceiver(name, workQueue, receiver, destinationID);
}

void StreamClientConnection::removeWorkQueueMessageReceiver(ReceiverName name, uint64_t destinationID)
{
    protectedConnection()->removeWorkQueueMessageReceiver(name, destinationID);
}

}
