/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "WebPageProxy.h"

#include "PageClient.h"
#include "WebKitWebResourceLoadManager.h"
#include "WebPageProxyInternals.h"
#include "WebPreferences.h"
#include <WebCore/NotImplemented.h>
#include <WebCore/SearchPopupMenu.h>
#include <WebCore/UserAgent.h>

namespace WebKit {

String WebPageProxy::userAgentForURL(const URL& url)
{
    if (url.isNull() || !preferences().needsSiteSpecificQuirks())
        return this->userAgent();

    auto userAgent = WebCore::standardUserAgentForURL(url);
    return userAgent.isNull() ? this->userAgent() : userAgent;
}

String WebPageProxy::standardUserAgent(const String& applicationNameForUserAgent)
{
    return WebCore::standardUserAgent(applicationNameForUserAgent);
}

void WebPageProxy::saveRecentSearches(IPC::Connection&, const String&, const Vector<WebCore::RecentSearch>&)
{
    notImplemented();
}

void WebPageProxy::loadRecentSearches(IPC::Connection&, const String&, CompletionHandler<void(Vector<WebCore::RecentSearch>&&)>&& completionHandler)
{
    notImplemented();
    completionHandler({ });
}

void WebPageProxy::didInitiateLoadForResource(WebCore::ResourceLoaderIdentifier resourceID, WebCore::FrameIdentifier frameID, WebCore::ResourceRequest&& request)
{
    if (auto* manager = pageClient() ? pageClient()->webResourceLoadManager() : nullptr)
        manager->didInitiateLoad(resourceID, frameID, WTFMove(request));
}

void WebPageProxy::didSendRequestForResource(WebCore::ResourceLoaderIdentifier resourceID, WebCore::FrameIdentifier frameID, WebCore::ResourceRequest&& request, WebCore::ResourceResponse&& redirectResponse)
{
    if (auto* manager = pageClient() ? pageClient()->webResourceLoadManager() : nullptr)
        manager->didSendRequest(resourceID, frameID, WTFMove(request), WTFMove(redirectResponse));
}

void WebPageProxy::didReceiveResponseForResource(WebCore::ResourceLoaderIdentifier resourceID, WebCore::FrameIdentifier frameID, WebCore::ResourceResponse&& response)
{
    if (auto* manager = pageClient() ? pageClient()->webResourceLoadManager() : nullptr)
        manager->didReceiveResponse(resourceID, frameID, WTFMove(response));
}

void WebPageProxy::didFinishLoadForResource(WebCore::ResourceLoaderIdentifier resourceID, WebCore::FrameIdentifier frameID, WebCore::ResourceError&& error)
{
    if (auto* manager = pageClient() ? pageClient()->webResourceLoadManager() : nullptr)
        manager->didFinishLoad(resourceID, frameID, WTFMove(error));
}

void WebPageProxy::scheduleActivityStateUpdate()
{
    if (internals().activityStateChangeTimer.isActive())
        return;

    internals().activityStateChangeTimer.startOneShot(0_s);
}

} // namespace WebKit
