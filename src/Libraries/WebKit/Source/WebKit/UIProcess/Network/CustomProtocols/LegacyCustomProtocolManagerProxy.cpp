/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include "LegacyCustomProtocolManagerProxy.h"

#include "APICustomProtocolManagerClient.h"
#include "LegacyCustomProtocolManagerMessages.h"
#include "LegacyCustomProtocolManagerProxyMessages.h"
#include "NetworkProcessProxy.h"
#include "WebProcessPool.h"
#include <WebCore/ResourceRequest.h>

namespace WebKit {

LegacyCustomProtocolManagerProxy::LegacyCustomProtocolManagerProxy(NetworkProcessProxy& networkProcessProxy)
    : m_networkProcessProxy(networkProcessProxy)
{
    networkProcessProxy.addMessageReceiver(Messages::LegacyCustomProtocolManagerProxy::messageReceiverName(), *this);
}

void LegacyCustomProtocolManagerProxy::ref() const
{
    m_networkProcessProxy->ref();
}

void LegacyCustomProtocolManagerProxy::deref() const
{
    m_networkProcessProxy->deref();
}

Ref<NetworkProcessProxy> LegacyCustomProtocolManagerProxy::protectedProcess()
{
    return m_networkProcessProxy.get();
}

LegacyCustomProtocolManagerProxy::~LegacyCustomProtocolManagerProxy()
{
    Ref<NetworkProcessProxy> networkProcessProxy = m_networkProcessProxy.get();
    networkProcessProxy->removeMessageReceiver(Messages::LegacyCustomProtocolManagerProxy::messageReceiverName());
    invalidate();
}

void LegacyCustomProtocolManagerProxy::startLoading(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceRequest& request)
{
    protectedProcess()->customProtocolManagerClient().startLoading(*this, customProtocolID, request);
}

void LegacyCustomProtocolManagerProxy::stopLoading(LegacyCustomProtocolID customProtocolID)
{
    protectedProcess()->customProtocolManagerClient().stopLoading(*this, customProtocolID);
}

void LegacyCustomProtocolManagerProxy::invalidate()
{
    Ref<NetworkProcessProxy> networkProcessProxy = m_networkProcessProxy.get();
    networkProcessProxy->customProtocolManagerClient().invalidate(*this);
}

void LegacyCustomProtocolManagerProxy::wasRedirectedToRequest(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceRequest& request, const WebCore::ResourceResponse& redirectResponse)
{
    protectedProcess()->send(Messages::LegacyCustomProtocolManager::WasRedirectedToRequest(customProtocolID, request, redirectResponse), 0);
}

void LegacyCustomProtocolManagerProxy::didReceiveResponse(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceResponse& response, CacheStoragePolicy cacheStoragePolicy)
{
    protectedProcess()->send(Messages::LegacyCustomProtocolManager::DidReceiveResponse(customProtocolID, response, cacheStoragePolicy), 0);
}

void LegacyCustomProtocolManagerProxy::didLoadData(LegacyCustomProtocolID customProtocolID, std::span<const uint8_t> data)
{
    protectedProcess()->send(Messages::LegacyCustomProtocolManager::DidLoadData(customProtocolID, data), 0);
}

void LegacyCustomProtocolManagerProxy::didFailWithError(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceError& error)
{
    protectedProcess()->send(Messages::LegacyCustomProtocolManager::DidFailWithError(customProtocolID, error), 0);
}

void LegacyCustomProtocolManagerProxy::didFinishLoading(LegacyCustomProtocolID customProtocolID)
{
    protectedProcess()->send(Messages::LegacyCustomProtocolManager::DidFinishLoading(customProtocolID), 0);
}

} // namespace WebKit
