/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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

#include "InjectedBundlePageContextMenuClient.h"

#include "APIArray.h"
#include "InjectedBundleHitTestResult.h"
#include "Logging.h"
#include "WebContextMenuItem.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebPage.h"
#include <WebCore/ContextMenu.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedBundlePageContextMenuClient);

InjectedBundlePageContextMenuClient::InjectedBundlePageContextMenuClient(const WKBundlePageContextMenuClientBase* client)
{
    initialize(client);
}

bool InjectedBundlePageContextMenuClient::getCustomMenuFromDefaultItems(WebPage& page, const HitTestResult& hitTestResult, const Vector<ContextMenuItem>& proposedMenu, Vector<WebContextMenuItemData>& newMenu, const WebCore::ContextMenuContext&, RefPtr<API::Object>& userData)
{
    if (!m_client.getContextMenuFromDefaultMenu)
        return false;

    auto defaultMenuItems = kitItems(proposedMenu).map([](auto& item) -> RefPtr<API::Object> {
        return WebContextMenuItem::create(item);
    });

    WKArrayRef newMenuWK = nullptr;
    WKTypeRef userDataToPass = nullptr;
    m_client.getContextMenuFromDefaultMenu(toAPI(&page), toAPI(&InjectedBundleHitTestResult::create(hitTestResult).get()), toAPI(API::Array::create(WTFMove(defaultMenuItems)).ptr()), &newMenuWK, &userDataToPass, m_client.base.clientInfo);
    RefPtr<API::Array> array = adoptRef(toImpl(newMenuWK));
    userData = adoptRef(toImpl(userDataToPass));
    
    newMenu.clear();
    
    if (!array || !array->size())
        return true;
    
    size_t size = array->size();
    for (size_t i = 0; i < size; ++i) {
        WebContextMenuItem* item = array->at<WebContextMenuItem>(i);
        if (!item) {
            LOG(ContextMenu, "New menu entry at index %i is not a WebContextMenuItem", (int)i);
            continue;
        }
        
        newMenu.append(item->data());
    }
    
    return true;
}

void InjectedBundlePageContextMenuClient::prepareForImmediateAction(WebPage& page, const HitTestResult& hitTestResult, RefPtr<API::Object>& userData)
{
    if (!m_client.prepareForActionMenu)
        return;

    WKTypeRef userDataToPass = nullptr;
    m_client.prepareForActionMenu(toAPI(&page), toAPI(&InjectedBundleHitTestResult::create(hitTestResult).get()), &userDataToPass, m_client.base.clientInfo);
    userData = adoptRef(toImpl(userDataToPass));
}

} // namespace WebKit
#endif // ENABLE(CONTEXT_MENUS)
