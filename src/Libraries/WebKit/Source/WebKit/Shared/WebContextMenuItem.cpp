/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

#if ENABLE(CONTEXT_MENUS)

#include "WebContextMenuItem.h"

#include "APIArray.h"
#include <WebCore/ContextMenuItem.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

WebContextMenuItem::WebContextMenuItem(const WebContextMenuItemData& data)
    : m_webContextMenuItemData(data)
{
}

Ref<WebContextMenuItem> WebContextMenuItem::create(const String& title, bool enabled, API::Array* submenuItems)
{
    size_t size = submenuItems->size();

    Vector<WebContextMenuItemData> submenu;
    submenu.reserveInitialCapacity(size);
    for (size_t i = 0; i < size; ++i) {
        if (auto* item = submenuItems->at<WebContextMenuItem>(i))
            submenu.append(item->data());
    }
    submenu.shrinkToFit();

    bool checked = false;
    unsigned indentationLevel = 0;
    return adoptRef(*new WebContextMenuItem(WebContextMenuItemData(WebCore::ContextMenuItemType::Submenu, WebCore::ContextMenuItemTagNoAction, String { title }, enabled, checked, indentationLevel, WTFMove(submenu)))).leakRef();
}

WebContextMenuItem* WebContextMenuItem::separatorItem()
{
    static NeverDestroyed<Ref<WebContextMenuItem>> separatorItem = adoptRef(*new WebContextMenuItem(WebContextMenuItemData(WebCore::ContextMenuItemType::Separator, WebCore::ContextMenuItemTagNoAction, String(), true, false)));
    return separatorItem->ptr();
}

Ref<API::Array> WebContextMenuItem::submenuItemsAsAPIArray() const
{
    if (m_webContextMenuItemData.type() != WebCore::ContextMenuItemType::Submenu)
        return API::Array::create();

    auto submenuItems = m_webContextMenuItemData.submenu().map([](auto& item) -> RefPtr<API::Object> {
        return WebContextMenuItem::create(item);
    });
    return API::Array::create(WTFMove(submenuItems));
}

API::Object* WebContextMenuItem::userData() const
{
    return m_webContextMenuItemData.userData();
}

void WebContextMenuItem::setUserData(API::Object* userData)
{
    m_webContextMenuItemData.setUserData(userData);
}

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
