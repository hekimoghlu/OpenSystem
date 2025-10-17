/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include "APIObject.h"
#include "APIViewClient.h"
#include "PageClientImpl.h"
#include "WKView.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class PlayStationWebView : public API::ObjectImpl<API::Object::Type::View> {
    WTF_MAKE_TZONE_ALLOCATED(PlayStationWebView);
public:
#if USE(WPE_BACKEND_PLAYSTATION)
    static RefPtr<PlayStationWebView> create(struct wpe_view_backend*, const API::PageConfiguration&);
#else
    static RefPtr<PlayStationWebView> create(const API::PageConfiguration&);
#endif
    virtual ~PlayStationWebView();

    void setClient(std::unique_ptr<API::ViewClient>&&);

    WebPageProxy* page() { return m_page.get(); }

    void setViewSize(WebCore::IntSize);
    WebCore::IntSize viewSize() const { return m_viewSize; }

    void setViewState(OptionSet<WebCore::ActivityState>);
    OptionSet<WebCore::ActivityState> viewState() const { return m_viewStateFlags; }

#if USE(WPE_BACKEND_PLAYSTATION)
    struct wpe_view_backend* backend() { return m_backend; }
#endif

#if ENABLE(FULLSCREEN_API)
    void willEnterFullScreen(CompletionHandler<void(bool)>&&);
    void didEnterFullScreen();
    void willExitFullScreen();
    void didExitFullScreen();
    void requestExitFullScreen();
#endif

    // Functions called by PageClientImpl
    void setViewNeedsDisplay(const WebCore::Region&);
#if ENABLE(FULLSCREEN_API)
    bool isFullScreen();
    void closeFullScreenManager();
    void enterFullScreen(CompletionHandler<void(bool)>&&);
    void exitFullScreen();
    void beganEnterFullScreen(const WebCore::IntRect&, const WebCore::IntRect&);
    void beganExitFullScreen(const WebCore::IntRect&, const WebCore::IntRect&);
#endif
    void setCursor(const WebCore::Cursor&);

private:
#if USE(WPE_BACKEND_PLAYSTATION)
    PlayStationWebView(struct wpe_view_backend*, const API::PageConfiguration&);
#else
    PlayStationWebView(const API::PageConfiguration&);
#endif

    std::unique_ptr<API::ViewClient> m_client;
    std::unique_ptr<WebKit::PageClientImpl> m_pageClient;
    RefPtr<WebPageProxy> m_page;
    OptionSet<WebCore::ActivityState> m_viewStateFlags;

    WebCore::IntSize m_viewSize;
#if USE(WPE_BACKEND_PLAYSTATION)
    struct wpe_view_backend* m_backend;
#endif
#if ENABLE(FULLSCREEN_API)
    bool m_isFullScreen { false };
#endif
};

} // namespace WebKit
