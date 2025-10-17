/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#include "WebKitWebResourceLoadManager.h"

#include "WebFrameProxy.h"
#include "WebKitWebResourcePrivate.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebKitWebResourceLoadManager);

WebKitWebResourceLoadManager::WebKitWebResourceLoadManager(WebKitWebView* webView)
    : m_webView(webView)
{
    g_signal_connect(m_webView, "load-changed", G_CALLBACK(+[](WebKitWebView*, WebKitLoadEvent loadEvent, WebKitWebResourceLoadManager* manager) {
        if (loadEvent == WEBKIT_LOAD_STARTED)
            manager->m_resources.clear();
    }), this);
}

WebKitWebResourceLoadManager::~WebKitWebResourceLoadManager()
{
    g_signal_handlers_disconnect_by_data(m_webView, this);
}

void WebKitWebResourceLoadManager::didInitiateLoad(ResourceLoaderIdentifier resourceID, FrameIdentifier frameID, ResourceRequest&& request)
{
    RefPtr frame = WebFrameProxy::webFrame(frameID);
    if (!frame)
        return;

    GRefPtr<WebKitWebResource> resource = adoptGRef(webkitWebResourceCreate(*frame, request));
    m_resources.set({ resourceID, frameID }, resource);
    webkitWebViewResourceLoadStarted(m_webView, resource.get(), WTFMove(request));
}

void WebKitWebResourceLoadManager::didSendRequest(ResourceLoaderIdentifier resourceID, FrameIdentifier frameID, ResourceRequest&& request, ResourceResponse&& redirectResponse)
{
    if (auto* resource = m_resources.get({ resourceID, frameID }))
        webkitWebResourceSentRequest(resource, WTFMove(request), WTFMove(redirectResponse));
}

void WebKitWebResourceLoadManager::didReceiveResponse(ResourceLoaderIdentifier resourceID, FrameIdentifier frameID, ResourceResponse&& response)
{
    if (auto* resource = m_resources.get({ resourceID, frameID }))
        webkitWebResourceSetResponse(resource, WTFMove(response));
}

void WebKitWebResourceLoadManager::didFinishLoad(ResourceLoaderIdentifier resourceID, FrameIdentifier frameID, ResourceError&& error)
{
    auto resource = m_resources.take({ resourceID, frameID });
    if (!resource)
        return;

    if (error.isNull())
        webkitWebResourceFinished(resource.get());
    else
        webkitWebResourceFailed(resource.get(), WTFMove(error));
}
} // namespace WebKit
