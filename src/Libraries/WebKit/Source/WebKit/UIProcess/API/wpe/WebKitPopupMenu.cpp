/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#include "WebKitPopupMenu.h"

#include "APIViewClient.h"
#include "WPEWebView.h"
#include "WebKitOptionMenuPrivate.h"
#include "WebKitWebViewClient.h"

namespace WebKit {
using namespace WebCore;

Ref<WebKitPopupMenu> WebKitPopupMenu::create(WKWPE::View& view, WebPopupMenuProxy::Client& client)
{
    ASSERT(view.client().isGLibBasedAPI());
    return adoptRef(*new WebKitPopupMenu(view, client));
}

WebKitPopupMenu::WebKitPopupMenu(WKWPE::View& view, WebPopupMenuProxy::Client& client)
    : WebPopupMenuProxy(client)
    , m_view(view)
{
}

static void menuCloseCallback(WebKitPopupMenu* popupMenu)
{
    popupMenu->activateItem(std::nullopt);
}

void WebKitPopupMenu::showPopupMenu(const IntRect& rect, TextDirection direction, double pageScaleFactor, const Vector<WebPopupItem>& items, const PlatformPopupMenuData& platformData, int32_t selectedIndex)
{
    GRefPtr<WebKitOptionMenu> menu = static_cast<WebKitWebViewClient&>(m_view.client()).showOptionMenu(*this, rect, items, selectedIndex);
    if (menu) {
        m_menu = WTFMove(menu);
        g_signal_connect_swapped(m_menu.get(), "close", G_CALLBACK(menuCloseCallback), this);
    }
}

void WebKitPopupMenu::hidePopupMenu()
{
    if (m_menu) {
        g_signal_handlers_disconnect_matched(m_menu.get(), G_SIGNAL_MATCH_DATA, 0, 0, nullptr, nullptr, this);
        webkit_option_menu_close(m_menu.get());
    }
}

void WebKitPopupMenu::cancelTracking()
{
    hidePopupMenu();
    m_menu = nullptr;
}

void WebKitPopupMenu::selectItem(unsigned itemIndex)
{
    if (CheckedPtr client = this->client())
        client->setTextFromItemForPopupMenu(this, itemIndex);
    m_selectedItem = itemIndex;
}

void WebKitPopupMenu::activateItem(std::optional<unsigned> itemIndex)
{
    if (CheckedPtr client = this->client())
        client->valueChangedForPopupMenu(this, itemIndex.value_or(m_selectedItem.value_or(-1)));
    if (m_menu) {
        g_signal_handlers_disconnect_matched(m_menu.get(), G_SIGNAL_MATCH_DATA, 0, 0, nullptr, nullptr, this);
        m_menu = nullptr;
    }
}

} // namespace WebKit
