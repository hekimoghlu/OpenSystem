/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include "LegacyCustomProtocolManager.h"

#include "LegacyCustomProtocolManagerMessages.h"
#include "LegacyCustomProtocolManagerProxyMessages.h"
#include "NetworkProcess.h"
#include "NetworkProcessCreationParameters.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyCustomProtocolManager);

ASCIILiteral LegacyCustomProtocolManager::supplementName()
{
    return "LegacyCustomProtocolManager"_s;
}

LegacyCustomProtocolManager::LegacyCustomProtocolManager(NetworkProcess& networkProcess)
    : m_networkProcess(networkProcess)
{
    m_networkProcess->addMessageReceiver(Messages::LegacyCustomProtocolManager::messageReceiverName(), *this);
}

void LegacyCustomProtocolManager::ref() const
{
    m_networkProcess->ref();
}

void LegacyCustomProtocolManager::deref() const
{
    m_networkProcess->deref();
}

Ref<NetworkProcess> LegacyCustomProtocolManager::protectedNetworkProcess() const
{
    ASSERT(RunLoop::isMain());
    return m_networkProcess.get();
}

void LegacyCustomProtocolManager::initialize(const NetworkProcessCreationParameters& parameters)
{
    registerProtocolClass();

    for (const auto& scheme : parameters.urlSchemesRegisteredForCustomProtocols)
        registerScheme(scheme);
}

LegacyCustomProtocolID LegacyCustomProtocolManager::addCustomProtocol(CustomProtocol&& customProtocol)
{
    Locker locker { m_customProtocolMapLock };
    auto customProtocolID = LegacyCustomProtocolID::generate();
    m_customProtocolMap.add(customProtocolID, WTFMove(customProtocol));
    return customProtocolID;
}

void LegacyCustomProtocolManager::removeCustomProtocol(LegacyCustomProtocolID customProtocolID)
{
    Locker locker { m_customProtocolMapLock };
    m_customProtocolMap.remove(customProtocolID);
}

void LegacyCustomProtocolManager::startLoading(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceRequest& request)
{
    protectedNetworkProcess()->send(Messages::LegacyCustomProtocolManagerProxy::StartLoading(customProtocolID, request));
}

void LegacyCustomProtocolManager::stopLoading(LegacyCustomProtocolID customProtocolID)
{
    protectedNetworkProcess()->send(Messages::LegacyCustomProtocolManagerProxy::StopLoading(customProtocolID), 0);
}

} // namespace WebKit
