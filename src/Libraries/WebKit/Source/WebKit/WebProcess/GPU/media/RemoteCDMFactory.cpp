/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "RemoteCDMFactory.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteCDM.h"
#include "RemoteCDMFactoryProxyMessages.h"
#include "RemoteCDMInstanceSession.h"
#include "WebProcess.h"
#include <WebCore/Settings.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteCDMFactory);

RemoteCDMFactory::RemoteCDMFactory(WebProcess& webProcess)
    : m_webProcess(webProcess)
{
}

RemoteCDMFactory::~RemoteCDMFactory() = default;

void RemoteCDMFactory::ref() const
{
    return m_webProcess->ref();
}

void RemoteCDMFactory::deref() const
{
    return m_webProcess->deref();
}

void RemoteCDMFactory::registerFactory(Vector<CDMFactory*>& factories)
{
    factories.append(this);
}

ASCIILiteral RemoteCDMFactory::supplementName()
{
    return "RemoteCDMFactory"_s;
}

GPUProcessConnection& RemoteCDMFactory::gpuProcessConnection()
{
    return WebProcess::singleton().ensureGPUProcessConnection();
}

bool RemoteCDMFactory::supportsKeySystem(const String& keySystem)
{
    auto sendResult = gpuProcessConnection().connection().sendSync(Messages::RemoteCDMFactoryProxy::SupportsKeySystem(keySystem), { });
    auto [supported] = sendResult.takeReplyOr(false);
    return supported;
}

std::unique_ptr<CDMPrivate> RemoteCDMFactory::createCDM(const String& keySystem, const CDMPrivateClient&)
{
    auto sendResult = gpuProcessConnection().connection().sendSync(Messages::RemoteCDMFactoryProxy::CreateCDM(keySystem), { });
    auto [identifier, configuration] = sendResult.takeReplyOr(std::nullopt, RemoteCDMConfiguration { });
    if (!identifier)
        return nullptr;
    return RemoteCDM::create(*this, WTFMove(*identifier), WTFMove(configuration));
}

void RemoteCDMFactory::addSession(RemoteCDMInstanceSession& session)
{
    ASSERT(!m_sessions.contains(session.identifier()));
    m_sessions.set(session.identifier(), session);
}

void RemoteCDMFactory::removeSession(RemoteCDMInstanceSessionIdentifier identifier)
{
    ASSERT(m_sessions.contains(identifier));
    m_sessions.remove(identifier);
    gpuProcessConnection().connection().send(Messages::RemoteCDMFactoryProxy::RemoveSession(identifier), { });
}

void RemoteCDMFactory::removeInstance(RemoteCDMInstanceIdentifier identifier)
{
    gpuProcessConnection().connection().send(Messages::RemoteCDMFactoryProxy::RemoveInstance(identifier), { });
}

void RemoteCDMFactory::didReceiveSessionMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (ObjectIdentifier<RemoteCDMInstanceSessionIdentifierType>::isValidIdentifier(decoder.destinationID())) {
        if (auto session = m_sessions.get(ObjectIdentifier<RemoteCDMInstanceSessionIdentifierType>(decoder.destinationID())))
            session->didReceiveMessage(connection, decoder);
    }
}

}

#endif
