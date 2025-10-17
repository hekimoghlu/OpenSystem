/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include "ContextMenuContextData.h"
#include "WKBase.h"
#include "WebContextMenuItem.h"
#include "WebContextMenuListenerProxy.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS NSMenu;

namespace WebCore {
class IntPoint;
}

namespace WebKit {
class WebContextMenuItemData;
class WebPageProxy;
}

namespace API {

class ContextMenuClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ContextMenuClient);
public:
    virtual ~ContextMenuClient() { }

    virtual void getContextMenuFromProposedMenu(WebKit::WebPageProxy&, Vector<Ref<WebKit::WebContextMenuItem>>&& proposedMenu, WebKit::WebContextMenuListenerProxy& listener, const WebKit::WebHitTestResultData&, API::Object* /* userData */) { listener.useContextMenuItems(WTFMove(proposedMenu)); }
    virtual void customContextMenuItemSelected(WebKit::WebPageProxy&, const WebKit::WebContextMenuItemData&) { }
    virtual void showContextMenu(WebKit::WebPageProxy&, const WebCore::IntPoint&, const Vector<Ref<WebKit::WebContextMenuItem>>&) { }
    virtual bool canShowContextMenu() const { return false; }
    virtual bool hideContextMenu(WebKit::WebPageProxy&) { return false; }

#if PLATFORM(MAC)
    virtual void menuFromProposedMenu(WebKit::WebPageProxy&, NSMenu *menu, const WebKit::ContextMenuContextData&, API::Object*, CompletionHandler<void(RetainPtr<NSMenu>&&)>&& completionHandler) { completionHandler(menu); }
#endif
};

} // namespace API

#endif // ENABLE(CONTEXT_MENUS)
