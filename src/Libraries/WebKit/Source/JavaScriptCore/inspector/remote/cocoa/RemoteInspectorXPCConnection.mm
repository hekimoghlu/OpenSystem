/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#import "RemoteInspectorXPCConnection.h"

#if ENABLE(REMOTE_INSPECTOR)

#import "JSCInlines.h"
#import <Foundation/Foundation.h>
#import <mutex>
#import <wtf/Assertions.h>
#import <wtf/Lock.h>
#import <wtf/Ref.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/spi/cocoa/CFXPCBridgeSPI.h>
#import <wtf/spi/cocoa/SecuritySPI.h>
#import <wtf/spi/darwin/XPCSPI.h>

namespace Inspector {

// Constants private to this file for message serialization on both ends.
#define RemoteInspectorXPCConnectionMessageNameKey @"messageName"
#define RemoteInspectorXPCConnectionUserInfoKey @"userInfo"
#define RemoteInspectorXPCConnectionSerializedMessageKey "msgData"

RemoteInspectorXPCConnection::RemoteInspectorXPCConnection(xpc_connection_t connection, dispatch_queue_t queue, Client* client)
    : m_connection(connection)
    , m_queue(queue)
    , m_client(client)
{
    xpc_connection_set_target_queue(m_connection.get(), m_queue.get());
    xpc_connection_set_event_handler(m_connection.get(), ^(xpc_object_t object) {
        handleEvent(object);
    });

    // Balanced by deref when the xpc_connection receives XPC_ERROR_CONNECTION_INVALID.
    ref();

    xpc_connection_resume(m_connection.get());
}

RemoteInspectorXPCConnection::~RemoteInspectorXPCConnection()
{
    ASSERT(!m_client);
    ASSERT(!m_connection);
    ASSERT(m_closed);
}

void RemoteInspectorXPCConnection::close()
{
    Locker locker { m_mutex };
    closeFromMessage();
}

void RemoteInspectorXPCConnection::closeFromMessage()
{
    m_closed = true;
    m_client = nullptr;

    dispatch_async(m_queue.get(), ^{
        Locker locker { m_mutex };
        // This will trigger one last XPC_ERROR_CONNECTION_INVALID event on the queue and deref us.
        closeOnQueue();
    });
}

void RemoteInspectorXPCConnection::closeOnQueue()
{
    if (m_connection) {
        xpc_connection_cancel(m_connection.get());
        m_connection = nullptr;
    }

    m_queue = nullptr;
}

NSDictionary *RemoteInspectorXPCConnection::deserializeMessage(xpc_object_t object)
{
    if (xpc_get_type(object) != XPC_TYPE_DICTIONARY)
        return nil;

    xpc_object_t xpcDictionary = xpc_dictionary_get_value(object, RemoteInspectorXPCConnectionSerializedMessageKey);
    if (!xpcDictionary || xpc_get_type(xpcDictionary) != XPC_TYPE_DICTIONARY) {
        Locker locker { m_mutex };
        if (m_client)
            m_client->xpcConnectionUnhandledMessage(this, object);
        return nil;
    }

    auto dictionary = dynamic_cf_cast<CFDictionaryRef>(adoptCF(_CFXPCCreateCFObjectFromXPCMessage(xpcDictionary)));
    ASSERT_WITH_MESSAGE(dictionary, "Unable to deserialize xpc message");
    return dictionary.bridgingAutorelease();
}

void RemoteInspectorXPCConnection::handleEvent(xpc_object_t object)
{
    if (xpc_get_type(object) == XPC_TYPE_ERROR) {
        {
            Locker locker { m_mutex };
            if (m_client)
                m_client->xpcConnectionFailed(this);

            m_closed = true;
            m_client = nullptr;
            closeOnQueue();
        }

        if (object == XPC_ERROR_CONNECTION_INVALID) {
            // This is the last event we will ever receive from the connection.
            // This balances the ref() in the constructor.
            deref();
        }
        return;
    }

#if PLATFORM(MAC)
    if (!m_validated) {
        audit_token_t token;
        xpc_connection_get_audit_token(m_connection.get(), &token);
        if (!WTF::hasEntitlement(token, "com.apple.private.webinspector.webinspectord"_s)) {
            Locker locker { m_mutex };
            // This will trigger one last XPC_ERROR_CONNECTION_INVALID event on the queue and deref us.
            closeOnQueue();
            return;
        }
        m_validated = true;
    }
#endif

    NSDictionary *dictionary = deserializeMessage(object);
    if (![dictionary isKindOfClass:[NSDictionary class]])
        return;

    NSString *message = dictionary[RemoteInspectorXPCConnectionMessageNameKey];
    if (![message isKindOfClass:[NSString class]])
        return;

    NSDictionary *userInfo = dictionary[RemoteInspectorXPCConnectionUserInfoKey];
    if (userInfo && ![userInfo isKindOfClass:[NSDictionary class]])
        return;

    Locker locker { m_mutex };
    if (m_client)
        m_client->xpcConnectionReceivedMessage(this, message, userInfo);
}

void RemoteInspectorXPCConnection::sendMessage(NSString *messageName, NSDictionary *userInfo)
{
    ASSERT(!m_closed);
    if (m_closed)
        return;

    auto dictionary = adoptNS([[NSMutableDictionary alloc] init]);
    [dictionary setObject:messageName forKey:RemoteInspectorXPCConnectionMessageNameKey];
    if (userInfo)
        [dictionary setObject:userInfo forKey:RemoteInspectorXPCConnectionUserInfoKey];

    auto xpcDictionary = adoptOSObject(_CFXPCCreateXPCMessageWithCFObject((__bridge CFDictionaryRef)dictionary.get()));
    ASSERT_WITH_MESSAGE(xpcDictionary && xpc_get_type(xpcDictionary.get()) == XPC_TYPE_DICTIONARY, "Unable to serialize xpc message");
    if (!xpcDictionary)
        return;

    auto msg = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_value(msg.get(), RemoteInspectorXPCConnectionSerializedMessageKey, xpcDictionary.get());
    xpc_connection_send_message(m_connection.get(), msg.get());
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
