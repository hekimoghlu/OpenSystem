/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#ifndef WebContextMenuItem_h
#define WebContextMenuItem_h

#if ENABLE(CONTEXT_MENUS)

#include "APIObject.h"
#include "WebContextMenuItemData.h"

namespace API {
class Array;
}

namespace WebCore {
class ContextMenuItem;
}

namespace WebKit {

class WebContextMenuItem : public API::ObjectImpl<API::Object::Type::ContextMenuItem> {
public:
    static Ref<WebContextMenuItem> create(const WebContextMenuItemData& data)
    {
        return adoptRef(*new WebContextMenuItem(data));
    }

    static Ref<WebContextMenuItem> create(const String& title, bool enabled, API::Array* submenuItems);
    static WebContextMenuItem* separatorItem();

    Ref<API::Array> submenuItemsAsAPIArray() const;

    API::Object* userData() const;
    void setUserData(API::Object*);

    const WebContextMenuItemData& data() { return m_webContextMenuItemData; }

private:
    WebContextMenuItem(const WebContextMenuItemData&);

    WebContextMenuItemData m_webContextMenuItemData;
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
#endif // WebContextMenuItem_h
