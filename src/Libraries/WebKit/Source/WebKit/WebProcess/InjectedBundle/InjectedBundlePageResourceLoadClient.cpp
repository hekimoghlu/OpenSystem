/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#include "InjectedBundlePageResourceLoadClient.h"

#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedBundlePageResourceLoadClient);

InjectedBundlePageResourceLoadClient::InjectedBundlePageResourceLoadClient(const WKBundlePageResourceLoadClientBase* client)
{
    initialize(client);
}

void InjectedBundlePageResourceLoadClient::didInitiateLoadForResource(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier, const ResourceRequest& request, bool pageIsProvisionallyLoading)
{
    if (!m_client.didInitiateLoadForResource)
        return;

    m_client.didInitiateLoadForResource(toAPI(&page), toAPI(&frame), identifier.toUInt64(), toAPI(request), pageIsProvisionallyLoading, m_client.base.clientInfo);
}

void InjectedBundlePageResourceLoadClient::willSendRequestForFrame(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier, ResourceRequest& request, const ResourceResponse& redirectResponse)
{
    if (!m_client.willSendRequestForFrame)
        return;

    RefPtr<API::URLRequest> returnedRequest = adoptRef(toImpl(m_client.willSendRequestForFrame(toAPI(&page), toAPI(&frame), identifier.toUInt64(), toAPI(request), toAPI(redirectResponse), m_client.base.clientInfo)));
    if (returnedRequest) {
        // If the client returned an HTTP body, we want to use that http body. This is needed to fix <rdar://problem/23763584>
        auto& returnedResourceRequest = returnedRequest->resourceRequest();
        RefPtr<FormData> returnedHTTPBody = returnedResourceRequest.httpBody();
        request.updateFromDelegatePreservingOldProperties(returnedResourceRequest);
        if (returnedHTTPBody)
            request.setHTTPBody(WTFMove(returnedHTTPBody));
    } else
        request = { };
}

void InjectedBundlePageResourceLoadClient::didReceiveResponseForResource(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier, const ResourceResponse& response)
{
    if (!m_client.didReceiveResponseForResource)
        return;

    m_client.didReceiveResponseForResource(toAPI(&page), toAPI(&frame), identifier.toUInt64(), toAPI(response), m_client.base.clientInfo);
}

void InjectedBundlePageResourceLoadClient::didReceiveContentLengthForResource(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier, uint64_t contentLength)
{
    if (!m_client.didReceiveContentLengthForResource)
        return;

    m_client.didReceiveContentLengthForResource(toAPI(&page), toAPI(&frame), identifier.toUInt64(), contentLength, m_client.base.clientInfo);
}

void InjectedBundlePageResourceLoadClient::didFinishLoadForResource(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier)
{
    if (!m_client.didFinishLoadForResource)
        return;

    m_client.didFinishLoadForResource(toAPI(&page), toAPI(&frame), identifier.toUInt64(), m_client.base.clientInfo);
}

void InjectedBundlePageResourceLoadClient::didFailLoadForResource(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier, const ResourceError& error)
{
    if (!m_client.didFailLoadForResource)
        return;

    m_client.didFailLoadForResource(toAPI(&page), toAPI(&frame), identifier.toUInt64(), toAPI(error), m_client.base.clientInfo);
}

bool InjectedBundlePageResourceLoadClient::shouldCacheResponse(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier)
{
    if (!m_client.shouldCacheResponse)
        return true;

    return m_client.shouldCacheResponse(toAPI(&page), toAPI(&frame), identifier.toUInt64(), m_client.base.clientInfo);
}

bool InjectedBundlePageResourceLoadClient::shouldUseCredentialStorage(WebPage& page, WebFrame& frame, WebCore::ResourceLoaderIdentifier identifier)
{
    if (!m_client.shouldUseCredentialStorage)
        return true;

    return m_client.shouldUseCredentialStorage(toAPI(&page), toAPI(&frame), identifier.toUInt64(), m_client.base.clientInfo);
}

} // namespace WebKit
