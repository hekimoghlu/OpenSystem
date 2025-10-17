/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "WebContextMenu.h"

#if ENABLE(CONTEXT_MENUS)

#include "APIInjectedBundlePageContextMenuClient.h"
#include "ContextMenuContextData.h"
#include "MessageSenderInlines.h"
#include "UserData.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/ContextMenu.h>
#include <WebCore/ContextMenuController.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/Page.h>

namespace WebKit {
using namespace WebCore;

WebContextMenu::WebContextMenu(WebPage& page)
    : m_page(page)
{
}

WebContextMenu::~WebContextMenu()
{
}

void WebContextMenu::show()
{
    ContextMenuController& controller = m_page->corePage()->contextMenuController();
    RefPtr frame = controller.hitTestResult().innerNodeFrame();
    if (!frame)
        return;
    RefPtr view = frame->view();
    if (!view)
        return;

    Vector<WebContextMenuItemData> menuItems;
    RefPtr<API::Object> userData;
    menuItemsWithUserData(menuItems, userData);

    auto menuLocation = view->contentsToRootView(controller.hitTestResult().roundedPointInInnerNodeFrame());

    ContextMenuContextData contextMenuContextData(menuLocation, menuItems, controller.context());

    m_page->showContextMenuFromFrame(frame->frameID(), contextMenuContextData, UserData(WebProcess::singleton().transformObjectsToHandles(userData.get()).get()));
}

void WebContextMenu::itemSelected(const WebContextMenuItemData& item)
{
    m_page->corePage()->contextMenuController().contextMenuItemSelected(static_cast<ContextMenuAction>(item.action()), item.title());
}

void WebContextMenu::menuItemsWithUserData(Vector<WebContextMenuItemData> &menuItems, RefPtr<API::Object>& userData) const
{
    ContextMenuController& controller = m_page->corePage()->contextMenuController();

    ContextMenu* menu = controller.contextMenu();
    if (!menu)
        return;

    // Give the bundle client a chance to process the menu.
    const Vector<ContextMenuItem>& coreItems = menu->items();

    RefPtr page = m_page.get();
    if (page->injectedBundleContextMenuClient().getCustomMenuFromDefaultItems(*page, controller.hitTestResult(), coreItems, menuItems, controller.context(), userData))
        return;
    menuItems = kitItems(coreItems);
}

Vector<WebContextMenuItemData> WebContextMenu::items() const
{
    Vector<WebContextMenuItemData> menuItems;
    RefPtr<API::Object> userData;
    menuItemsWithUserData(menuItems, userData);
    return menuItems;
}

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
