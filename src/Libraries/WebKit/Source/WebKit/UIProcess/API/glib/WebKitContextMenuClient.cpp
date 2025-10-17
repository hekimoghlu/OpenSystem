/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#include "WebKitContextMenuClient.h"

#if ENABLE(CONTEXT_MENUS)
#include "APIContextMenuClient.h"
#include "APIString.h"
#include "WebContextMenuItem.h"
#include "WebKitWebViewPrivate.h"

using namespace WebKit;

class ContextMenuClient final: public API::ContextMenuClient {
public:
    explicit ContextMenuClient(WebKitWebView* webView)
        : m_webView(webView)
    {
    }

private:
    void getContextMenuFromProposedMenu(WebPageProxy&, Vector<Ref<WebKit::WebContextMenuItem>>&& proposedMenu, WebKit::WebContextMenuListenerProxy& contextMenuListener, const WebHitTestResultData& hitTestResultData, API::Object* userData) override
    {
        GRefPtr<GVariant> variant;
        if (userData) {
            CString userDataString = downcast<API::String>(userData)->string().utf8();
            WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port
            variant = adoptGRef(g_variant_parse(nullptr, userDataString.data(), userDataString.data() + userDataString.length(), nullptr, nullptr));
            WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        }

        auto menuItems = WTF::map(proposedMenu, [](auto& item) {
            return item->data();
        });
        webkitWebViewPopulateContextMenu(m_webView, menuItems, hitTestResultData, variant.get());
        contextMenuListener.useContextMenuItems({ });
    }

    WebKitWebView* m_webView;
};

void attachContextMenuClientToView(WebKitWebView* webView)
{
    webkitWebViewGetPage(webView).setContextMenuClient(makeUnique<ContextMenuClient>(webView));
}

#endif // ENABLE(CONTEXT_MENUS)
