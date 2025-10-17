/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "WebSearchPopupMenu.h"

#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <wtf/text/AtomString.h>

namespace WebKit {
using namespace WebCore;

Ref<WebSearchPopupMenu> WebSearchPopupMenu::create(WebPage* page, PopupMenuClient* client)
{
    return adoptRef(*new WebSearchPopupMenu(page, client));
}

WebSearchPopupMenu::WebSearchPopupMenu(WebPage* page, PopupMenuClient* client)
    : m_popup(WebPopupMenu::create(page, client))
{
}

PopupMenu* WebSearchPopupMenu::popupMenu()
{
    return m_popup.get();
}

RefPtr<WebPopupMenu> WebSearchPopupMenu::protectedPopup()
{
    return m_popup;
}

void WebSearchPopupMenu::saveRecentSearches(const AtomString& name, const Vector<RecentSearch>& searchItems)
{
    if (name.isEmpty())
        return;

    RefPtr page = protectedPopup()->page();
    if (!page)
        return;

    WebProcess::singleton().protectedParentProcessConnection()->send(Messages::WebPageProxy::SaveRecentSearches(name, searchItems), page->identifier());
}

void WebSearchPopupMenu::loadRecentSearches(const AtomString& name, Vector<RecentSearch>& resultItems)
{
    if (name.isEmpty())
        return;

    RefPtr page = protectedPopup()->page();
    if (!page)
        return;

    auto sendResult = WebProcess::singleton().protectedParentProcessConnection()->sendSync(Messages::WebPageProxy::LoadRecentSearches(name), page->identifier());
    if (sendResult.succeeded())
        std::tie(resultItems) = sendResult.takeReply();
}

bool WebSearchPopupMenu::enabled()
{
    return true;
}

} // namespace WebKit
