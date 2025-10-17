/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

#include "WebContextMenuListenerProxy.h"
#include "WebContextMenuProxy.h"

namespace WebKit {

class WebContextMenuItemData;
class WebPageProxy;

class WebContextMenuProxyWin final : public WebContextMenuProxy {
public:
    static auto create(WebPageProxy& page, ContextMenuContextData&& context, const UserData& userData)
    {
        return adoptRef(*new WebContextMenuProxyWin(page, WTFMove(context), userData));
    }
    ~WebContextMenuProxyWin();

private:
    WebContextMenuProxyWin(WebPageProxy&, ContextMenuContextData&&, const UserData&);
    void showContextMenuWithItems(Vector<Ref<WebContextMenuItem>>&&) override;

    HMENU m_menu;
};


} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
