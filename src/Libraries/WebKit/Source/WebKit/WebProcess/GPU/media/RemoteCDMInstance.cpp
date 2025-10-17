/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "RemoteCDMInstance.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteCDMInstanceMessages.h"
#include "RemoteCDMInstanceProxyMessages.h"
#include "RemoteCDMInstanceSession.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include "WebProcess.h"
#include <WebCore/CDMKeySystemConfiguration.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteCDMInstance> RemoteCDMInstance::create(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMInstanceIdentifier&& identifier, RemoteCDMInstanceConfiguration&& configuration)
{
    return adoptRef(*new RemoteCDMInstance(WTFMove(factory), WTFMove(identifier), WTFMove(configuration)));
}

RemoteCDMInstance::RemoteCDMInstance(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMInstanceIdentifier&& identifier, RemoteCDMInstanceConfiguration&& configuration)
    : m_factory(WTFMove(factory))
    , m_identifier(WTFMove(identifier))
    , m_configuration(WTFMove(configuration))
{
    if (RefPtr factory = m_factory.get())
        factory->gpuProcessConnection().messageReceiverMap().addMessageReceiver(Messages::RemoteCDMInstance::messageReceiverName(), m_identifier.toUInt64(), *this);
}

RemoteCDMInstance::~RemoteCDMInstance()
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return;

    factory->removeInstance(m_identifier);
    factory->gpuProcessConnection().messageReceiverMap().removeMessageReceiver(Messages::RemoteCDMInstance::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteCDMInstance::unrequestedInitializationDataReceived(const String& type, Ref<SharedBuffer>&& initData)
{
    if (RefPtr client = m_client.get())
        client->unrequestedInitializationDataReceived(type, WTFMove(initData));
}

void RemoteCDMInstance::initializeWithConfiguration(const WebCore::CDMKeySystemConfiguration& configuration, AllowDistinctiveIdentifiers distinctiveIdentifiers, AllowPersistentState persistentState, SuccessCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(Failed);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceProxy::InitializeWithConfiguration(configuration, distinctiveIdentifiers, persistentState), WTFMove(callback), m_identifier);
}

void RemoteCDMInstance::setServerCertificate(Ref<WebCore::SharedBuffer>&& certificate, SuccessCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(Failed);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceProxy::SetServerCertificate(WTFMove(certificate)), WTFMove(callback), m_identifier);
}

void RemoteCDMInstance::setStorageDirectory(const String& directory)
{
    if (RefPtr factory = m_factory.get())
        factory->gpuProcessConnection().connection().send(Messages::RemoteCDMInstanceProxy::SetStorageDirectory(directory), m_identifier);
}

RefPtr<WebCore::CDMInstanceSession> RemoteCDMInstance::createSession()
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return nullptr;

    uint64_t logIdentifier { 0 };
#if !RELEASE_LOG_DISABLED
    if (m_client)
        logIdentifier = reinterpret_cast<uint64_t>(m_client->logIdentifier());
#endif

    auto sendResult = factory->gpuProcessConnection().connection().sendSync(Messages::RemoteCDMInstanceProxy::CreateSession(logIdentifier), m_identifier);
    auto [identifier] = sendResult.takeReplyOr(std::nullopt);
    if (!identifier)
        return nullptr;
    auto session = RemoteCDMInstanceSession::create(factory.get(), WTFMove(*identifier));
    factory->addSession(session.copyRef());
    return session;
}

}

#endif
