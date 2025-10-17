/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#if ENABLE(MODEL_PROCESS)

#include "Connection.h"
#include "MessageReceiverMap.h"
#include <wtf/AbstractThreadSafeRefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class WebPage;
struct ModelProcessConnectionInfo;
struct WebPageCreationParameters;


class ModelProcessConnection
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ModelProcessConnection>
    , public IPC::Connection::Client {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ModelProcessConnection);
public:
    static RefPtr<ModelProcessConnection> create(IPC::Connection& parentConnection);
    ~ModelProcessConnection();

    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    IPC::Connection& connection() { return m_connection.get(); }
    IPC::MessageReceiverMap& messageReceiverMap() { return m_messageReceiverMap; }

#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> auditToken();
#endif

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    void createVisibilityPropagationContextForPage(WebPage&);
    void destroyVisibilityPropagationContextForPage(WebPage&);
#endif

    void configureLoggingChannel(const String&, WTFLogChannelState, WTFLogLevel);

    class Client : public AbstractThreadSafeRefCountedAndCanMakeWeakPtr {
    public:
        virtual ~Client() = default;

        virtual void modelProcessConnectionDidClose(ModelProcessConnection&) { }
    };
    void addClient(const Client& client) { m_clients.add(client); }

    static constexpr Seconds defaultTimeout = 3_s;

private:
    ModelProcessConnection(IPC::Connection::Identifier&&);
    bool waitForDidInitialize();
    void invalidate();

    // IPC::Connection::Client
    void didClose(IPC::Connection&) override;
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t indexOfObjectFailingDecoding) override;

    bool dispatchMessage(IPC::Connection&, IPC::Decoder&);
    bool dispatchSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    // Messages.
    void didInitialize(std::optional<ModelProcessConnectionInfo>&&);

    // The connection from the web process to the model process.
    Ref<IPC::Connection> m_connection;
    IPC::MessageReceiverMap m_messageReceiverMap;
    bool m_hasInitialized { false };
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> m_auditToken;
#endif

    ThreadSafeWeakHashSet<Client> m_clients;
};

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
