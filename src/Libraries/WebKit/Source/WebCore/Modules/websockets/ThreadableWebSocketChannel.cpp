/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include "ThreadableWebSocketChannel.h"

#include "ContentRuleListResults.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "HTTPHeaderValues.h"
#include "Page.h"
#include "Quirks.h"
#include "ScriptExecutionContext.h"
#include "SocketProvider.h"
#include "ThreadableWebSocketChannelClientWrapper.h"
#include "UserContentProvider.h"
#include "WebSocketChannelClient.h"
#include "WorkerGlobalScope.h"
#include "WorkerRunLoop.h"
#include "WorkerThread.h"
#include "WorkerThreadableWebSocketChannel.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

RefPtr<ThreadableWebSocketChannel> ThreadableWebSocketChannel::create(Document& document, WebSocketChannelClient& client, SocketProvider& provider)
{
    return provider.createWebSocketChannel(document, client);
}

RefPtr<ThreadableWebSocketChannel> ThreadableWebSocketChannel::create(ScriptExecutionContext& context, WebSocketChannelClient& client, SocketProvider& provider)
{
    if (RefPtr workerGlobalScope = dynamicDowncast<WorkerGlobalScope>(context)) {
        WorkerRunLoop& runLoop = workerGlobalScope->thread().runLoop();
        return WorkerThreadableWebSocketChannel::create(*workerGlobalScope, client, makeString("webSocketChannelMode"_s, runLoop.createUniqueId()), provider);
    }

    return create(downcast<Document>(context), client, provider);
}

ThreadableWebSocketChannel::ThreadableWebSocketChannel() = default;

std::optional<ThreadableWebSocketChannel::ValidatedURL> ThreadableWebSocketChannel::validateURL(Document& document, const URL& requestedURL)
{
    ValidatedURL validatedURL { requestedURL, true };
    if (RefPtr page = document.page()) {
        if (!page->allowsLoadFromURL(requestedURL, MainFrameMainResource::No))
            return { };
#if ENABLE(CONTENT_EXTENSIONS)
        if (RefPtr documentLoader = document.loader()) {
            auto results = page->protectedUserContentProvider()->processContentRuleListsForLoad(*page, validatedURL.url, ContentExtensions::ResourceType::WebSocket, *documentLoader);
            if (results.summary.blockedLoad)
                return { };
            if (results.summary.madeHTTPS) {
                ASSERT(validatedURL.url.protocolIs("ws"_s));
                validatedURL.url.setProtocol("wss"_s);
            }
            validatedURL.areCookiesAllowed = !results.summary.blockedCookies;
        }
#else
        UNUSED_PARAM(document);
#endif
    }
    return validatedURL;
}

std::optional<ResourceRequest> ThreadableWebSocketChannel::webSocketConnectRequest(Document& document, const URL& url)
{
    auto validatedURL = validateURL(document, url);
    if (!validatedURL)
        return { };

    ResourceRequest request { validatedURL->url };
    request.setHTTPUserAgent(document.userAgent(validatedURL->url));
    request.setDomainForCachePartition(document.domainForCachePartition());
    request.setAllowCookies(validatedURL->areCookiesAllowed);
    request.setFirstPartyForCookies(document.firstPartyForCookies());
    request.setHTTPHeaderField(HTTPHeaderName::Origin, document.securityOrigin().toString());

    if (RefPtr documentLoader = document.loader())
        request.setIsAppInitiated(documentLoader->lastNavigationWasAppInitiated());

    FrameLoader::addSameSiteInfoToRequestIfNeeded(request, &document);

    // Add no-cache headers to avoid compatibility issue.
    // There are some proxies that rewrite "Connection: upgrade"
    // to "Connection: close" in the response if a request doesn't contain
    // these headers.
    request.addHTTPHeaderField(HTTPHeaderName::Pragma, HTTPHeaderValues::noCache());
    request.addHTTPHeaderField(HTTPHeaderName::CacheControl, HTTPHeaderValues::noCache());

    auto httpURL = request.url();
    httpURL.setProtocol(url.protocolIs("ws"_s) ? "http"_s : "https"_s);
    auto requestOrigin = SecurityOrigin::create(httpURL);
    if (requestOrigin->isPotentiallyTrustworthy() && !document.quirks().shouldDisableFetchMetadata()) {
        request.addHTTPHeaderField(HTTPHeaderName::SecFetchDest, "websocket"_s);
        request.addHTTPHeaderField(HTTPHeaderName::SecFetchMode, "websocket"_s);

        if (document.protectedSecurityOrigin()->isSameOriginAs(requestOrigin.get()))
            request.addHTTPHeaderField(HTTPHeaderName::SecFetchSite, "same-origin"_s);
        else if (document.protectedSecurityOrigin()->isSameSiteAs(requestOrigin))
            request.addHTTPHeaderField(HTTPHeaderName::SecFetchSite, "same-site"_s);
        else
            request.addHTTPHeaderField(HTTPHeaderName::SecFetchSite, "cross-site"_s);
    }

    return request;
}

} // namespace WebCore
