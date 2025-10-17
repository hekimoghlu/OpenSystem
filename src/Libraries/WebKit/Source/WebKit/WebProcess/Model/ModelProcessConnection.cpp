/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#include "ModelProcessConnection.h"

#if ENABLE(MODEL_PROCESS)

#include "Logging.h"
#include "ModelConnectionToWebProcessMessages.h"
#include "ModelProcessConnectionInfo.h"
#include "ModelProcessConnectionMessages.h"
#include "ModelProcessConnectionParameters.h"
#include "ModelProcessModelPlayerManager.h"
#include "ModelProcessModelPlayerMessages.h"
#include "WebPage.h"
#include "WebPageCreationParameters.h"
#include "WebPageMessages.h"
#include "WebProcess.h"
#include "WebProcessProxyMessages.h"

namespace WebKit {
using namespace WebCore;

static ModelProcessConnectionParameters getModelProcessConnectionParameters()
{
    ModelProcessConnectionParameters parameters;
#if PLATFORM(COCOA)
    parameters.webProcessIdentity = ProcessIdentity { ProcessIdentity::CurrentProcess };
#endif
    return parameters;
}

RefPtr<ModelProcessConnection> ModelProcessConnection::create(IPC::Connection& parentConnection)
{
    auto connectionIdentifiers = IPC::Connection::createConnectionIdentifierPair();
    if (!connectionIdentifiers)
        return nullptr;

    parentConnection.send(Messages::WebProcessProxy::CreateModelProcessConnection(WTFMove(connectionIdentifiers->client), getModelProcessConnectionParameters()), 0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);

    auto instance = adoptRef(*new ModelProcessConnection(WTFMove(connectionIdentifiers->server)));
#if ENABLE(IPC_TESTING_API)
    if (parentConnection.ignoreInvalidMessageForTesting())
        instance->connection().setIgnoreInvalidMessageForTesting();
#endif
    return instance;
}

ModelProcessConnection::ModelProcessConnection(IPC::Connection::Identifier&& connectionIdentifier)
    : m_connection(IPC::Connection::createServerConnection(WTFMove(connectionIdentifier)))
{
    m_connection->open(*this);
}

ModelProcessConnection::~ModelProcessConnection()
{
    m_connection->invalidate();
}

#if HAVE(AUDIT_TOKEN)
std::optional<audit_token_t> ModelProcessConnection::auditToken()
{
    if (!waitForDidInitialize())
        return std::nullopt;
    return m_auditToken;
}
#endif

void ModelProcessConnection::invalidate()
{
    m_connection->invalidate();
    m_hasInitialized = true;
}

void ModelProcessConnection::didClose(IPC::Connection&)
{
    RELEASE_LOG(Process, "%p - ModelProcessConnection::didClose", this);
    auto protector = Ref { *this };
    WebProcess::singleton().modelProcessConnectionClosed(*this);

    m_clients.forEach([this] (auto& client) {
        client.modelProcessConnectionDidClose(*this);
    });
    m_clients.clear();
}

void ModelProcessConnection::didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t)
{
}

bool ModelProcessConnection::dispatchMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (decoder.messageReceiverName() == Messages::ModelProcessModelPlayer::messageReceiverName()) {
        WebProcess::singleton().modelProcessModelPlayerManager().didReceivePlayerMessage(connection, decoder);
        return true;
    }

    return false;
}

bool ModelProcessConnection::dispatchSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    return messageReceiverMap().dispatchSyncMessage(connection, decoder, replyEncoder);
}

void ModelProcessConnection::didInitialize(std::optional<ModelProcessConnectionInfo>&& info)
{
    if (!info) {
        invalidate();
        return;
    }
    m_hasInitialized = true;
}

bool ModelProcessConnection::waitForDidInitialize()
{
    if (!m_hasInitialized) {
        auto result = m_connection->waitForAndDispatchImmediately<Messages::ModelProcessConnection::DidInitialize>(0, defaultTimeout);
        if (result != IPC::Error::NoError) {
            RELEASE_LOG_ERROR(Process, "%p - ModelProcessConnection::waitForDidInitialize - failed, error:%" PUBLIC_LOG_STRING, this, IPC::errorAsString(result).characters());
            invalidate();
            return false;
        }
    }
    return m_connection->isValid();
}

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
void ModelProcessConnection::createVisibilityPropagationContextForPage(WebPage& page)
{
    connection().send(Messages::ModelConnectionToWebProcess::CreateVisibilityPropagationContextForPage(page.webPageProxyIdentifier(), page.identifier(), page.canShowWhileLocked()), { });
}

void ModelProcessConnection::destroyVisibilityPropagationContextForPage(WebPage& page)
{
    connection().send(Messages::ModelConnectionToWebProcess::DestroyVisibilityPropagationContextForPage(page.webPageProxyIdentifier(), page.identifier()), { });
}
#endif

void ModelProcessConnection::configureLoggingChannel(const String& channelName, WTFLogChannelState state, WTFLogLevel level)
{
    connection().send(Messages::ModelConnectionToWebProcess::ConfigureLoggingChannel(channelName, state, level), { });
}

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
