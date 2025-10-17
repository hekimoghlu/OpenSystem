/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

#include "EditorState.h"
#include "WebsiteDataStore.h"
#include <WebCore/NotImplemented.h>
#include <WebCore/SearchPopupMenu.h>
#include <WebCore/UserAgent.h>

#if USE(GRAPHICS_LAYER_WC)
#include "PageClientImpl.h"
#endif

namespace WebKit {

void WebPageProxy::platformInitialize()
{
    notImplemented();
}

struct wpe_view_backend* WebPageProxy::viewBackend()
{
    return nullptr;
}

String WebPageProxy::userAgentForURL(const URL&)
{
    return userAgent();
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

void WebPageProxy::didUpdateEditorState(const EditorState&, const EditorState&)
{
}

#if USE(GRAPHICS_LAYER_WC)
uint64_t WebPageProxy::viewWidget()
{
    return static_cast<PageClientImpl&>(pageClient()).viewWidget();
}
#endif

} // namespace WebKit
