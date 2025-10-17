/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include "NavigatorBeacon.h"

#include "CachedRawResource.h"
#include "CachedResourceLoader.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "HTTPParsers.h"
#include "LocalFrame.h"
#include "Navigator.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorBeacon);

NavigatorBeacon::NavigatorBeacon(Navigator& navigator)
    : m_navigator(navigator)
{
}

NavigatorBeacon::~NavigatorBeacon()
{
    for (auto& beacon : m_inflightBeacons)
        beacon->removeClient(*this);
}

NavigatorBeacon* NavigatorBeacon::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorBeacon*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorBeacon>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorBeacon::supplementName()
{
    return "NavigatorBeacon"_s;
}

void NavigatorBeacon::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    if (!resource.resourceError().isNull())
        logError(resource.resourceError());

    resource.removeClient(*this);
    bool wasRemoved = m_inflightBeacons.removeFirst(&resource);
    ASSERT_UNUSED(wasRemoved, wasRemoved);
    ASSERT(!m_inflightBeacons.contains(&resource));
}

void NavigatorBeacon::logError(const ResourceError& error)
{
    ASSERT(!error.isNull());

    RefPtr frame = m_navigator->frame();
    if (!frame)
        return;

    RefPtr document = frame->document();
    if (!document)
        return;

    ASCIILiteral messageMiddle { ". "_s };
    String description = error.localizedDescription();
    if (description.isEmpty()) {
        if (error.isAccessControl())
            messageMiddle = " due to access control checks."_s;
        else
            messageMiddle = "."_s;
    }

    document->addConsoleMessage(MessageSource::Network, MessageLevel::Error, makeString("Beacon API cannot load "_s, error.failingURL().string(), messageMiddle, description));
}

ExceptionOr<bool> NavigatorBeacon::sendBeacon(Document& document, const String& url, std::optional<FetchBody::Init>&& body)
{
    URL parsedUrl = document.completeURL(url);

    // Set parsedUrl to the result of the URL parser steps with url and base. If the algorithm returns an error, or if
    // parsedUrl's scheme is not "http" or "https", throw a "TypeError" exception and terminate these steps.
    if (!parsedUrl.isValid())
        return Exception { ExceptionCode::TypeError, "This URL is invalid"_s };
    if (!parsedUrl.protocolIsInHTTPFamily())
        return Exception { ExceptionCode::TypeError, "Beacons can only be sent over HTTP(S)"_s };

    if (!document.frame())
        return false;

    if (!document.shouldBypassMainWorldContentSecurityPolicy() && !document.checkedContentSecurityPolicy()->allowConnectToSource(parsedUrl)) {
        // We simulate a network error so we return true here. This is consistent with Blink.
        return true;
    }

    ResourceRequest request(parsedUrl);
    request.setHTTPMethod("POST"_s);
    request.setRequester(ResourceRequestRequester::Beacon);
    if (RefPtr documentLoader = document.loader())
        request.setIsAppInitiated(documentLoader->lastNavigationWasAppInitiated());

    ResourceLoaderOptions options;
    options.credentials = FetchOptions::Credentials::Include;
    options.cache = FetchOptions::Cache::NoCache;
    options.keepAlive = true;
    options.sendLoadCallbacks = SendCallbackPolicy::SendCallbacks;

    if (body) {
        options.mode = FetchOptions::Mode::NoCors;
        String mimeType;
        auto result = FetchBody::extract(WTFMove(body.value()), mimeType);
        if (result.hasException())
            return result.releaseException();
        auto fetchBody = result.releaseReturnValue();
        if (fetchBody.isReadableStream())
            return Exception { ExceptionCode::TypeError, "Beacons cannot send ReadableStream body"_s };

        request.setHTTPBody(fetchBody.bodyAsFormData());
        if (!mimeType.isEmpty()) {
            request.setHTTPContentType(mimeType);
            if (!isCrossOriginSafeRequestHeader(HTTPHeaderName::ContentType, mimeType)) {
                options.mode = FetchOptions::Mode::Cors;
                options.httpHeadersToKeep.add(HTTPHeadersToKeepFromCleaning::ContentType);
            }
        }
    }

    auto cachedResource = document.protectedCachedResourceLoader()->requestBeaconResource({ WTFMove(request), options });
    if (!cachedResource) {
        logError(cachedResource.error());
        return false;
    }

    ASSERT(!m_inflightBeacons.contains(cachedResource.value().get()));
    m_inflightBeacons.append(cachedResource.value().get());
    cachedResource.value()->addClient(*this);
    return true;
}

ExceptionOr<bool> NavigatorBeacon::sendBeacon(Navigator& navigator, Document& document, const String& url, std::optional<FetchBody::Init>&& body)
{
    return NavigatorBeacon::from(navigator)->sendBeacon(document, url, WTFMove(body));
}

}

