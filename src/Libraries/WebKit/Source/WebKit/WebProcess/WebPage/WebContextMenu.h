/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#include "WebContextMenuItemData.h"

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class Image;
}

namespace WebKit {

class WebPage;

class WebContextMenu : public RefCounted<WebContextMenu> {
public:
    static Ref<WebContextMenu> create(WebPage& page)
    {
        return adoptRef(*new WebContextMenu(page));
    }
    
    ~WebContextMenu();

    void show();
    void itemSelected(const WebContextMenuItemData&);
    Vector<WebContextMenuItemData> items() const;

private:
    WebContextMenu(WebPage&);
    void menuItemsWithUserData(Vector<WebContextMenuItemData>&, RefPtr<API::Object>&) const;

    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
