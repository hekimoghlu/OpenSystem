/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include "NetworkSessionCurl.h"

#include "AuthenticationManager.h"
#include "NetworkProcess.h"
#include "NetworkSessionCreationParameters.h"
#include "WebCookieManager.h"
#include "WebSocketTaskCurl.h"
#include <WebCore/CookieJarDB.h>
#include <WebCore/CurlContext.h>
#include <WebCore/NetworkStorageSession.h>

namespace WebKit {

using namespace WebCore;

NetworkSessionCurl::NetworkSessionCurl(NetworkProcess& networkProcess, const NetworkSessionCreationParameters& parameters)
    : NetworkSession(networkProcess, parameters)
{
    if (auto* storageSession = networkStorageSession()) {
        if (!parameters.cookiePersistentStorageFile.isEmpty())
            storageSession->setCookieDatabase(makeUniqueRef<CookieJarDB>(parameters.cookiePersistentStorageFile));
        storageSession->setProxySettings(parameters.proxySettings);
    }

    m_resourceLoadStatisticsDirectory = parameters.resourceLoadStatisticsParameters.directory;
    m_shouldIncludeLocalhostInResourceLoadStatistics = parameters.resourceLoadStatisticsParameters.shouldIncludeLocalhost ? ShouldIncludeLocalhost::Yes : ShouldIncludeLocalhost::No;
    m_enableResourceLoadStatisticsDebugMode = parameters.resourceLoadStatisticsParameters.enableDebugMode ? EnableResourceLoadStatisticsDebugMode::Yes : EnableResourceLoadStatisticsDebugMode::No;
    m_resourceLoadStatisticsManualPrevalentResource = parameters.resourceLoadStatisticsParameters.manualPrevalentResource;
    setTrackingPreventionEnabled(parameters.resourceLoadStatisticsParameters.enabled);
}

NetworkSessionCurl::~NetworkSessionCurl()
{

}

void NetworkSessionCurl::clearAlternativeServices(WallTime)
{
    if (auto* storageSession = networkStorageSession())
        storageSession->clearAlternativeServices();
}

std::unique_ptr<WebSocketTask> NetworkSessionCurl::createWebSocketTask(WebPageProxyIdentifier webPageProxyID, std::optional<FrameIdentifier>, std::optional<PageIdentifier>, NetworkSocketChannel& channel, const WebCore::ResourceRequest& request, const String& protocol, const WebCore::ClientOrigin& clientOrigin, bool, bool, OptionSet<WebCore::AdvancedPrivacyProtections>, StoredCredentialsPolicy)
{
    return makeUnique<WebSocketTask>(channel, webPageProxyID, request, protocol, clientOrigin);
}

void NetworkSessionCurl::didReceiveChallenge(WebSocketTask& webSocketTask, WebCore::AuthenticationChallenge&& challenge, CompletionHandler<void(WebKit::AuthenticationChallengeDisposition, const WebCore::Credential&)>&& challengeCompletionHandler)
{
    networkProcess().protectedAuthenticationManager()->didReceiveAuthenticationChallenge(sessionID(), webSocketTask.webPageProxyID(), !webSocketTask.topOrigin().isNull() ? &webSocketTask.topOrigin() : nullptr, challenge, NegotiatedLegacyTLS::No, WTFMove(challengeCompletionHandler));
}

} // namespace WebKit
