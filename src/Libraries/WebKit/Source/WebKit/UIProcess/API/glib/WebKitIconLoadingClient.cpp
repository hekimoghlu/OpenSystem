/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#include "WebKitIconLoadingClient.h"

#include "APIIconLoadingClient.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GWeakPtr.h>

using namespace WebKit;

class IconLoadingClient : public API::IconLoadingClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(IconLoadingClient);
public:
    explicit IconLoadingClient(WebKitWebView* webView)
        : m_webView(webView)
    {
    }

private:
    void getLoadDecisionForIcon(const WebCore::LinkIcon& icon, CompletionHandler<void(CompletionHandler<void(API::Data*)>&&)>&& completionHandler) override
    {
        // WebCore can send non HTTP icons.
        if (!icon.url.protocolIsInHTTPFamily()) {
            completionHandler(nullptr);
            return;
        }

        WebCore::LinkIcon copiedIcon = icon;
        webkitWebViewGetLoadDecisionForIcon(m_webView, icon, [weakWebView = GWeakPtr<WebKitWebView>(m_webView), icon = WTFMove(copiedIcon), completionHandler = WTFMove(completionHandler)] (bool loadIcon) mutable {
            if (!weakWebView || !loadIcon) {
                completionHandler(nullptr);
                return;
            }

            completionHandler([weakWebView = WTFMove(weakWebView), icon = WTFMove(icon)] (API::Data* iconData) {
                if (!weakWebView || !iconData)
                    return;
                webkitWebViewSetIcon(weakWebView.get(), icon, *iconData);
            });
        });
    }

    WebKitWebView* m_webView;
};

void attachIconLoadingClientToView(WebKitWebView* webView)
{
    webkitWebViewGetPage(webView).setIconLoadingClient(makeUnique<IconLoadingClient>(webView));
}
