/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

#include "WebContextMenuItemData.h"

#include "APIObject.h"
#include "ArgumentCoders.h"
#include <wtf/text/CString.h>
#include <WebCore/ContextMenu.h>

namespace WebKit {
using namespace WebCore;

WebContextMenuItemData::WebContextMenuItemData()
    : m_type(WebCore::ContextMenuItemType::Action)
    , m_action(WebCore::ContextMenuItemTagNoAction)
    , m_enabled(true)
    , m_checked(false)
    , m_indentationLevel(0)
{
}

WebContextMenuItemData::WebContextMenuItemData(WebCore::ContextMenuItemType type, WebCore::ContextMenuAction action, String&& title, bool enabled, bool checked, unsigned indentationLevel, Vector<WebContextMenuItemData>&& submenu)
    : m_type(type)
    , m_action(action)
    , m_title(WTFMove(title))
    , m_enabled(enabled)
    , m_checked(checked)
    , m_indentationLevel(indentationLevel)
    , m_submenu(WTFMove(submenu))
{
}

WebContextMenuItemData::WebContextMenuItemData(const WebCore::ContextMenuItem& item)
    : m_type(item.type())
    , m_action(item.action())
    , m_title(item.title())
{
    if (m_type == WebCore::ContextMenuItemType::Submenu) {
        const Vector<WebCore::ContextMenuItem>& coreSubmenu = item.subMenuItems();
        m_submenu = kitItems(coreSubmenu);
    }
    
    m_enabled = item.enabled();
    m_checked = item.checked();
    m_indentationLevel = item.indentationLevel();
}

ContextMenuItem WebContextMenuItemData::core() const
{
    if (m_type != ContextMenuItemType::Submenu)
        return ContextMenuItem(m_type, m_action, m_title, m_enabled, m_checked, m_indentationLevel);
    
    Vector<ContextMenuItem> subMenuItems = coreItems(m_submenu);
    return ContextMenuItem(m_action, m_title, m_enabled, m_checked, subMenuItems, m_indentationLevel);
}

API::Object* WebContextMenuItemData::userData() const
{
    return m_userData.get();
}

void WebContextMenuItemData::setUserData(API::Object* userData)
{
    m_userData = userData;
}

Vector<WebContextMenuItemData> kitItems(const Vector<WebCore::ContextMenuItem>& coreItemVector)
{
    return coreItemVector.map([](auto& item) {
        return WebContextMenuItemData { item };
    });
}

Vector<ContextMenuItem> coreItems(const Vector<WebContextMenuItemData>& kitItemVector)
{
    return kitItemVector.map([](auto& item) {
        return item.core();
    });
}

} // namespace WebKit
#endif // ENABLE(CONTEXT_MENUS)
