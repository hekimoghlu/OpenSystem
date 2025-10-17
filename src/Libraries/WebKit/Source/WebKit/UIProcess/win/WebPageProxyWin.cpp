/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#include "NativeWebKeyboardEvent.h"
#include "PageClientImpl.h"
#include "WebPageProxyInternals.h"
#include <WebCore/SearchPopupMenuDB.h>
#include <WebCore/UserAgent.h>

namespace WebKit {

void WebPageProxy::platformInitialize()
{
}

String WebPageProxy::userAgentForURL(const URL&)
{
    return userAgent();
}

String WebPageProxy::standardUserAgent(const String& applicationNameForUserAgent)
{
    return WebCore::standardUserAgent(applicationNameForUserAgent);
}

void WebPageProxy::saveRecentSearches(IPC::Connection&, const String& name, const Vector<WebCore::RecentSearch>& searchItems)
{
    if (!name)
        return;

    return WebCore::SearchPopupMenuDB::singleton().saveRecentSearches(name, searchItems);
}

void WebPageProxy::loadRecentSearches(IPC::Connection&, const String& name, CompletionHandler<void(Vector<WebCore::RecentSearch>&&)>&& completionHandler)
{
    if (!name)
        return completionHandler({ });

    Vector<WebCore::RecentSearch> searchItems;
    WebCore::SearchPopupMenuDB::singleton().loadRecentSearches(name, searchItems);
    completionHandler(WTFMove(searchItems));
}

void WebPageProxy::didUpdateEditorState(const EditorState&, const EditorState&)
{
}

#if USE(GRAPHICS_LAYER_TEXTURE_MAPPER) || USE(GRAPHICS_LAYER_WC)
uint64_t WebPageProxy::viewWidget()
{
    return reinterpret_cast<uint64_t>(static_cast<PageClientImpl&>(*pageClient()).viewWidget());
}
#endif

void WebPageProxy::dispatchPendingCharEvents(const NativeWebKeyboardEvent& keydownEvent)
{
    auto& pendingCharEvents = keydownEvent.pendingCharEvents();
    for (auto it = pendingCharEvents.rbegin(); it != pendingCharEvents.rend(); it++)
        internals().keyEventQueue.prepend(NativeWebKeyboardEvent(it->hwnd, it->message, it->wParam, it->lParam, { }));
}

} // namespace WebKit
