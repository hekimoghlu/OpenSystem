/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#include "WebContextMenuProxy.h"

#if ENABLE(CONTEXT_MENUS)

#include "APIContextMenuClient.h"
#include "MessageSenderInlines.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"

namespace WebKit {

WebContextMenuProxy::WebContextMenuProxy(WebPageProxy& page, ContextMenuContextData&& context, const UserData& userData)
    : m_context(WTFMove(context))
    , m_userData(userData)
    , m_page(page)
{
}

WebContextMenuProxy::~WebContextMenuProxy() = default;

Vector<Ref<WebContextMenuItem>> WebContextMenuProxy::proposedItems() const
{
    return WTF::map(m_context.menuItems(), [](auto& item) {
        return WebContextMenuItem::create(item);
    });
}

void WebContextMenuProxy::show()
{
    ASSERT(m_context.webHitTestResultData());

    RefPtr page = this->page();
    if (!page)
        return;

    m_contextMenuListener = WebContextMenuListenerProxy::create(*this);
    page->contextMenuClient().getContextMenuFromProposedMenu(*page, proposedItems(), *m_contextMenuListener, m_context.webHitTestResultData().value(),
        page->legacyMainFrameProcess().transformHandlesToObjects(m_userData.protectedObject().get()).get());
}

void WebContextMenuProxy::useContextMenuItems(Vector<Ref<WebContextMenuItem>>&& items)
{
    m_contextMenuListener = nullptr;

    RefPtr page = this->page();
    if (!page)
        return;

    // Since showContextMenuWithItems can spin a nested run loop we need to turn off the responsiveness timer.
    page->legacyMainFrameProcess().stopResponsivenessTimer();

    // Protect |this| from being deallocated if WebPageProxy code is re-entered from the menu runloop or delegates.
    Ref protectedThis { *this };
    showContextMenuWithItems(WTFMove(items));
    page->clearWaitingForContextMenuToShow();
}

} // namespace WebKit

#endif
