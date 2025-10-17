/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

#include "APIViewClient.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

typedef struct _WebKitWebView WebKitWebView;

namespace WKWPE {
class View;
}

namespace WebCore {
class IntRect;
}

namespace WebKit {
class WebKitPopupMenu;
class WebKitWebResourceLoadManager;
struct WebPopupItem;
struct UserMessage;
}

class WebKitWebViewClient final : public API::ViewClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebKitWebViewClient);
public:
    explicit WebKitWebViewClient(WebKitWebView*);

    GRefPtr<WebKitOptionMenu> showOptionMenu(WebKit::WebKitPopupMenu&, const WebCore::IntRect&, const Vector<WebKit::WebPopupItem>&, int32_t selectedIndex);

private:
    bool isGLibBasedAPI() override { return true; }

    void frameDisplayed(WKWPE::View&) override;
    void willStartLoad(WKWPE::View&) override;
    void didChangePageID(WKWPE::View&) override;
    void didReceiveUserMessage(WKWPE::View&, WebKit::UserMessage&&, CompletionHandler<void(WebKit::UserMessage&&)>&&) override;
    WebKit::WebKitWebResourceLoadManager* webResourceLoadManager() override;

#if ENABLE(FULLSCREEN_API)
    bool enterFullScreen(WKWPE::View&) override;
    bool exitFullScreen(WKWPE::View&) override;
#endif

    WebKitWebView* m_webView;
};
