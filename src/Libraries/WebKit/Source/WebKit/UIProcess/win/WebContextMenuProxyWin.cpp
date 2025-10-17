/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "WebContextMenuProxyWin.h"

#if ENABLE(CONTEXT_MENUS)

#include "APIContextMenuClient.h"
#include "WebContextMenuItem.h"
#include "WebContextMenuItemData.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"

namespace WebKit {
using namespace WebCore;

static HMENU createMenu(const ContextMenuContextData &context)
{
    HMENU menu = ::CreatePopupMenu();
    MENUINFO menuInfo;
    menuInfo.cbSize = sizeof(menuInfo);
    menuInfo.fMask = MIM_STYLE;
    menuInfo.dwStyle = MNS_NOTIFYBYPOS;
    menuInfo.dwMenuData = (ULONG_PTR)&context;
    ::SetMenuInfo(menu, &menuInfo);
    return menu;
}

static void populate(const ContextMenuContextData &, HMENU, const Vector<WebContextMenuItemData>&);

static void createMenuItem(const ContextMenuContextData &context, HMENU menu, const WebContextMenuItemData &data)
{
    UINT flags = 0;

    flags |= data.enabled() ? MF_ENABLED : MF_DISABLED;
    flags |= data.checked() ? MF_CHECKED : MF_UNCHECKED;

    switch (data.type()) {
    case ContextMenuItemType::Action:
    case ContextMenuItemType::CheckableAction:
        ::AppendMenu(menu, flags | MF_STRING, data.action(), data.title().wideCharacters().data());
        break;
    case ContextMenuItemType::Separator:
        ::AppendMenu(menu, flags | MF_SEPARATOR, data.action(), nullptr);
        break;
    case ContextMenuItemType::Submenu:
        HMENU submenu = createMenu(context);
        populate(context, submenu, data.submenu());
        ::AppendMenu(menu, flags | MF_POPUP, (UINT_PTR)submenu, data.title().wideCharacters().data());
        break;
    }
}

static void populate(const ContextMenuContextData &context, HMENU menu, const Vector<WebContextMenuItemData>& items)
{
    for (auto& data : items)
        createMenuItem(context, menu, data);
}

static void populate(const ContextMenuContextData &context, HMENU menu, const Vector<Ref<WebContextMenuItem>>& items)
{
    for (auto& item : items) {
        auto data = item->data();
        createMenuItem(context, menu, data);
    }
}

void WebContextMenuProxyWin::showContextMenuWithItems(Vector<Ref<WebContextMenuItem>>&& items)
{
    populate(m_context, m_menu, items);

    UINT flags = TPM_RIGHTBUTTON | TPM_TOPALIGN | TPM_VERPOSANIMATION | TPM_HORIZONTAL | TPM_LEFTALIGN | TPM_HORPOSANIMATION;
    auto location = m_context.menuLocation();
    location.scale(page()->deviceScaleFactor());
    POINT pt = location;
    HWND wnd = reinterpret_cast<HWND>(page()->viewWidget());
    ::ClientToScreen(wnd, &pt);
    ::TrackPopupMenuEx(m_menu, flags, pt.x, pt.y, wnd, nullptr);
}

WebContextMenuProxyWin::WebContextMenuProxyWin(WebPageProxy& page, ContextMenuContextData&& context, const UserData& userData)
    : WebContextMenuProxy(page, WTFMove(context), userData)
{
    m_menu = createMenu(m_context);
}

WebContextMenuProxyWin::~WebContextMenuProxyWin()
{
    if (m_menu)
        ::DestroyMenu(m_menu);
}

} // namespace WebKit
#endif // ENABLE(CONTEXT_MENUS)
