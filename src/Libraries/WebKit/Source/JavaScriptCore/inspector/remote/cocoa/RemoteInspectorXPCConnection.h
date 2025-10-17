/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

#import <dispatch/dispatch.h>
#import <wtf/Lock.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/ThreadSafeRefCounted.h>
#import <wtf/spi/darwin/XPCSPI.h>

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace Inspector {

class RemoteInspectorXPCConnection : public ThreadSafeRefCounted<RemoteInspectorXPCConnection> {
public:
    class Client {
    public:
        virtual ~Client() { }
        virtual void xpcConnectionReceivedMessage(RemoteInspectorXPCConnection*, NSString *messageName, NSDictionary *userInfo) = 0;
        virtual void xpcConnectionFailed(RemoteInspectorXPCConnection*) = 0;
        virtual void xpcConnectionUnhandledMessage(RemoteInspectorXPCConnection*, xpc_object_t) = 0;
    };

    RemoteInspectorXPCConnection(xpc_connection_t, dispatch_queue_t, Client*);
    virtual ~RemoteInspectorXPCConnection();

    void close();
    void closeFromMessage();
    void sendMessage(NSString *messageName, NSDictionary *userInfo);

private:
    NSDictionary *deserializeMessage(xpc_object_t);
    void handleEvent(xpc_object_t);
    void closeOnQueue();

    // We handle XPC events on the queue, but a client may call close() on any queue.
    // We make sure that m_client is thread safe and immediately cleared in close().
    Lock m_mutex;

    OSObjectPtr<xpc_connection_t> m_connection;
    OSObjectPtr<dispatch_queue_t> m_queue;
    Client* m_client;
    bool m_closed { false };
#if PLATFORM(MAC)
    bool m_validated { false };
#endif
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
