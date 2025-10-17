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
#import "config.h"
#import "DaemonConnection.h"

#import "DaemonEncoder.h"
#import "PrivateClickMeasurementConnection.h"
#import "WebPushDaemonConnection.h"
#import <wtf/BlockPtr.h>
#import <wtf/RunLoop.h>

namespace WebKit {

namespace Daemon {

void Connection::send(xpc_object_t message) const
{
    ASSERT(RunLoop::isMain());
    initializeConnectionIfNeeded();

    ASSERT(m_connection.get());
    ASSERT(xpc_get_type(message) == XPC_TYPE_DICTIONARY);
    xpc_connection_send_message(m_connection.get(), message);
}

void Connection::sendWithReply(xpc_object_t message, CompletionHandler<void(xpc_object_t)>&& completionHandler) const
{
    ASSERT(RunLoop::isMain());
    initializeConnectionIfNeeded();

    ASSERT(m_connection.get());
    ASSERT(xpc_get_type(message) == XPC_TYPE_DICTIONARY);
    xpc_connection_send_message_with_reply(m_connection.get(), message, dispatch_get_main_queue(), makeBlockPtr([completionHandler = WTFMove(completionHandler)] (xpc_object_t reply) mutable {
        ASSERT(RunLoop::isMain());
        completionHandler(reply);
    }).get());
}

template<typename Traits>
void ConnectionToMachService<Traits>::initializeConnectionIfNeeded() const
{
    if (m_connection)
        return;
    m_connection = adoptOSObject(xpc_connection_create_mach_service(m_machServiceName.data(), dispatch_get_main_queue(), 0));
    xpc_connection_set_event_handler(m_connection.get(), [weakThis = WeakPtr { *this }](xpc_object_t event) {
        if (!weakThis)
            return;
        if (event == XPC_ERROR_CONNECTION_INVALID) {
#if HAVE(XPC_CONNECTION_COPY_INVALIDATION_REASON)
            auto reason = std::unique_ptr<char[]>(xpc_connection_copy_invalidation_reason(weakThis->m_connection.get()));
            WTFLogAlways("Failed to connect to mach service %s, reason: %s", weakThis->m_machServiceName.data(), reason.get());
#else
            WTFLogAlways("Failed to connect to mach service %s, likely because it is not registered with launchd", weakThis->m_machServiceName.data());
#endif
        }
        if (event == XPC_ERROR_CONNECTION_INTERRUPTED) {
            // Daemon crashed, we will need to make a new connection to a new instance of the daemon.
            weakThis->m_connection = nullptr;
        }
        weakThis->connectionReceivedEvent(event);
    });
    xpc_connection_activate(m_connection.get());

    newConnectionWasInitialized();
}

template<typename Traits>
void ConnectionToMachService<Traits>::send(typename Traits::MessageType messageType, EncodedMessage&& message) const
{
    initializeConnectionIfNeeded();
    Connection::send(dictionaryFromMessage(messageType, WTFMove(message)).get());
}

template<typename Traits>
void ConnectionToMachService<Traits>::sendWithReply(typename Traits::MessageType messageType, EncodedMessage&& message, CompletionHandler<void(EncodedMessage&&)>&& completionHandler) const
{
    ASSERT(RunLoop::isMain());
    initializeConnectionIfNeeded();

    Connection::sendWithReply(dictionaryFromMessage(messageType, WTFMove(message)).get(), [completionHandler = WTFMove(completionHandler)] (xpc_object_t reply) mutable {
        if (xpc_get_type(reply) != XPC_TYPE_DICTIONARY) {
            ASSERT_NOT_REACHED();
            return completionHandler({ });
        }
        if (xpc_dictionary_get_uint64(reply, Traits::protocolVersionKey) != Traits::protocolVersionValue) {
            ASSERT_NOT_REACHED();
            return completionHandler({ });
        }
        completionHandler(xpc_dictionary_get_data_span(reply, Traits::protocolEncodedMessageKey));
    });
}

template class ConnectionToMachService<PCM::ConnectionTraits>;

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
template class ConnectionToMachService<WebPushD::ConnectionTraits>;
#endif

} // namespace Daemon

} // namespace WebKit
