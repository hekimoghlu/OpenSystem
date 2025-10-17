/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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

#include "WebPopupMenuProxy.h"
#include <wtf/glib/GRefPtr.h>

typedef struct _WebKitOptionMenu WebKitOptionMenu;

namespace WKWPE {
class View;
}

namespace WebKit {

class WebKitPopupMenu final : public WebPopupMenuProxy {
public:
    static Ref<WebKitPopupMenu> create(WKWPE::View&, WebPopupMenuProxy::Client&);
    ~WebKitPopupMenu() = default;

    void selectItem(unsigned itemIndex);
    void activateItem(std::optional<unsigned> itemIndex);

private:
    WebKitPopupMenu(WKWPE::View&, WebPopupMenuProxy::Client&);

    void showPopupMenu(const WebCore::IntRect&, WebCore::TextDirection, double pageScaleFactor, const Vector<WebPopupItem>&, const PlatformPopupMenuData&, int32_t selectedIndex) override;
    void hidePopupMenu() override;
    void cancelTracking() override;

    WKWPE::View& m_view;
    GRefPtr<WebKitOptionMenu> m_menu;
    std::optional<unsigned> m_selectedItem;
};

} // namespace WebKit
