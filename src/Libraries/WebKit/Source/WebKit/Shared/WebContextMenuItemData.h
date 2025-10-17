/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#pragma once

#if ENABLE(CONTEXT_MENUS)

#include <WebCore/ContextMenuItem.h>
#include <wtf/text/WTFString.h>

namespace API {
class Object;
}

namespace WebKit {

class WebContextMenuItemData {
public:
    WebContextMenuItemData();
    WebContextMenuItemData(const WebCore::ContextMenuItem&);
    WebContextMenuItemData(WebCore::ContextMenuItemType, WebCore::ContextMenuAction, String&& title, bool enabled, bool checked, unsigned indentationLevel = 0, Vector<WebContextMenuItemData>&& submenu = { });

    WebCore::ContextMenuItemType type() const { return m_type; }
    WebCore::ContextMenuAction action() const { return m_action; }
    const String& title() const { return m_title; }
    bool enabled() const { return m_enabled; }
    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool checked() const { return m_checked; }
    unsigned indentationLevel() const { return m_indentationLevel; }
    const Vector<WebContextMenuItemData>& submenu() const { return m_submenu; }
    
    WebCore::ContextMenuItem core() const;
    
    API::Object* userData() const;
    void setUserData(API::Object*);

private:
    WebCore::ContextMenuItemType m_type;
    WebCore::ContextMenuAction m_action;
    String m_title;
    bool m_enabled;
    bool m_checked;
    unsigned m_indentationLevel;
    Vector<WebContextMenuItemData> m_submenu;
    RefPtr<API::Object> m_userData;
};

Vector<WebContextMenuItemData> kitItems(const Vector<WebCore::ContextMenuItem>&);
Vector<WebCore::ContextMenuItem> coreItems(const Vector<WebContextMenuItemData>&);

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
