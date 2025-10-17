/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include <wtf/CompletionHandler.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/CString.h>

#if PLATFORM(COCOA)
#include <wtf/OSObjectPtr.h>
#include <wtf/spi/darwin/XPCSPI.h>
#endif

namespace WebKit {

namespace Daemon {

using EncodedMessage = Vector<uint8_t>;

class Connection : public RefCountedAndCanMakeWeakPtr<Connection> {
public:
#if PLATFORM(COCOA)
    static Ref<Connection> create(OSObjectPtr<xpc_connection_t>&& connection)
    {
        return adoptRef(*new Connection(WTFMove(connection)));
    }
#endif

    virtual ~Connection() = default;

#if PLATFORM(COCOA)
    xpc_connection_t get() const { return m_connection.get(); }
    void send(xpc_object_t) const;
    void sendWithReply(xpc_object_t, CompletionHandler<void(xpc_object_t)>&&) const;
#endif

protected:
    Connection() = default;

#if PLATFORM(COCOA)
    explicit Connection(OSObjectPtr<xpc_connection_t>&& connection)
        : m_connection(WTFMove(connection)) { }
#endif

    virtual void initializeConnectionIfNeeded() const { }

#if PLATFORM(COCOA)
    mutable OSObjectPtr<xpc_connection_t> m_connection;
#endif
};

template<typename Traits>
class ConnectionToMachService : public Connection {
public:
    virtual ~ConnectionToMachService() = default;

    void send(typename Traits::MessageType, EncodedMessage&&) const;
    void sendWithReply(typename Traits::MessageType, EncodedMessage&&, CompletionHandler<void(EncodedMessage&&)>&&) const;

    virtual void newConnectionWasInitialized() const = 0;

#if PLATFORM(COCOA)
    virtual OSObjectPtr<xpc_object_t> dictionaryFromMessage(typename Traits::MessageType, EncodedMessage&&) const = 0;
    virtual void connectionReceivedEvent(xpc_object_t) = 0;
#endif

    const CString& machServiceName() const { return m_machServiceName; }

protected:
    explicit ConnectionToMachService(CString&& machServiceName)
        : m_machServiceName(WTFMove(machServiceName))
    { }

private:
    void initializeConnectionIfNeeded() const final;

    const CString m_machServiceName;
};

} // namespace Daemon

} // namespace WebKit
