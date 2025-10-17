/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#include "EarlyHintsResourceLoader.h"

#include "MessageSenderInlines.h"
#include "NetworkLoadParameters.h"
#include "PreconnectTask.h"
#include "WebPageMessages.h"
#include <WebCore/ContentSecurityPolicy.h>
#include <WebCore/LinkHeader.h>
#include <WebCore/ResourceLoaderOptions.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/StoredCredentialsPolicy.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(EarlyHintsResourceLoader);

EarlyHintsResourceLoader::EarlyHintsResourceLoader(NetworkResourceLoader& loader)
    : m_loader(&loader)
{
}

EarlyHintsResourceLoader::~EarlyHintsResourceLoader() = default;

void EarlyHintsResourceLoader::addConsoleMessage(MessageSource messageSource, MessageLevel messageLevel, const String& message, unsigned long)
{
    if (!m_loader)
        return;

    m_loader->send(Messages::WebPage::AddConsoleMessage { m_loader->frameID(), messageSource, messageLevel, message, m_loader->coreIdentifier() }, m_loader->pageID());
}

void EarlyHintsResourceLoader::enqueueSecurityPolicyViolationEvent(SecurityPolicyViolationEventInit&&)
{
}

void EarlyHintsResourceLoader::handleEarlyHintsResponse(ResourceResponse&& response)
{
    RELEASE_ASSERT(response.httpStatusCode() == 103);

    if (!m_loader)
        return;

    // For consistency with other browsers, only process early hints for top-level navigation from
    // secure origins using HTTP/2 or later.
    if (!m_loader->isMainFrameLoad() || response.url().protocol() != "https"_s || response.httpVersion().startsWith("HTTP/1"_s))
        return;

    // Only the first early hint response served during the navigation is handled.
    // FIXME: discard hints on cross-origin redirect once we support early hint preloads.
    if (m_hasReceivedEarlyHints)
        return;
    m_hasReceivedEarlyHints = true;

    auto headerValue = response.httpHeaderField(HTTPHeaderName::Link);
    if (headerValue.isEmpty())
        return;

    auto url = response.url();
    ContentSecurityPolicy contentSecurityPolicy { URL { url }, this, nullptr };
    contentSecurityPolicy.didReceiveHeaders(ContentSecurityPolicyResponseHeaders { response }, m_loader->originalRequest().httpReferrer());

    LinkHeaderSet headerSet(headerValue);
    for (const auto& header : headerSet) {
        if (!header.valid() || header.url().isEmpty() || header.rel().isEmpty() || header.isViewportDependent())
            continue;

        if (equalLettersIgnoringASCIICase(header.rel(), "preconnect"_s))
            startPreconnectTask(response.url(), header, contentSecurityPolicy);
    }
}

ResourceRequest EarlyHintsResourceLoader::constructPreconnectRequest(const ResourceRequest& originalRequest, const URL& url)
{
    ResourceRequest request { url };

    // firstPartyForCookies and user agent are part of the HTTP socket pool keys in CFNetwork: rdar://59434166
    auto firstPartyForCookies = originalRequest.firstPartyForCookies();
    if (firstPartyForCookies.isValid())
        request.setFirstPartyForCookies(firstPartyForCookies);

    auto userAgent = originalRequest.httpUserAgent();
    if (!userAgent.isEmpty())
        request.setHTTPUserAgent(userAgent);

    return request;
}

void EarlyHintsResourceLoader::startPreconnectTask(const URL& baseURL, const LinkHeader& header, const ContentSecurityPolicy& contentSecurityPolicy)
{
#if ENABLE(SERVER_PRECONNECT)
    RefPtr loader = m_loader.get();
    if (!loader || !loader->parameters().linkPreconnectEarlyHintsEnabled)
        return;

    URL url(baseURL, header.url());
    if (!url.isValid() || url.protocol() != "https"_s)
        return;

    const auto& originalRequest = loader->originalRequest();
    if (!contentSecurityPolicy.allowConnectToSource(url, ContentSecurityPolicy::RedirectResponseReceived::No, originalRequest.url()))
        return;

    auto* networkSession = loader->protectedConnectionToWebProcess()->networkSession();
    if (!networkSession)
        return;

    NetworkLoadParameters parameters;
    auto globalFrameID = m_loader->globalFrameID();
    parameters.webPageProxyID = globalFrameID.webPageProxyID;
    parameters.webPageID = globalFrameID.webPageID;
    parameters.webFrameID = globalFrameID.frameID;
    parameters.storedCredentialsPolicy = equalLettersIgnoringASCIICase(header.crossOrigin(), "anonymous"_s) ? StoredCredentialsPolicy::DoNotUse : StoredCredentialsPolicy::Use;
    parameters.contentSniffingPolicy = ContentSniffingPolicy::DoNotSniffContent;
    parameters.contentEncodingSniffingPolicy = ContentEncodingSniffingPolicy::Default;
    parameters.shouldPreconnectOnly = PreconnectOnly::Yes;
    parameters.request = constructPreconnectRequest(originalRequest, url);
    parameters.isNavigatingToAppBoundDomain = m_loader->parameters().isNavigatingToAppBoundDomain;
    (new PreconnectTask(*networkSession, WTFMove(parameters), [](const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&) { }))->start();

    addConsoleMessage(MessageSource::Network, MessageLevel::Info, makeString("Preconnecting to "_s, url.string(), " due to early hint"_s));
#else
    UNUSED_PARAM(baseURL);
    UNUSED_PARAM(header);
    UNUSED_PARAM(contentSecurityPolicy);
#endif
}

} // namespace WebKit
