/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#include "WKView.h"

#include "APIClient.h"
#include "APIPageConfiguration.h"
#include "APIViewClient.h"
#include "PlayStationWebView.h"
#include "WKAPICast.h"
#include "WKSharedAPICast.h"
#include <WebCore/Cursor.h>
#include <WebCore/Region.h>

namespace API {
template<> struct ClientTraits<WKViewClientBase> {
    typedef std::tuple<WKViewClientV0> Versions;
};
}

WKCursorType toWKCursorType(const WebCore::Cursor& cursor)
{
    switch (cursor.type()) {
    case WebCore::Cursor::Type::Hand:
        return kWKCursorTypeHand;
    case WebCore::Cursor::Type::None:
        return kWKCursorTypeNone;
    case WebCore::Cursor::Type::Pointer:
    default:
        return kWKCursorTypePointer;
    }
}

WKViewRef WKViewCreate(WKPageConfigurationRef configuration)
{
#if USE(WPE_BACKEND_PLAYSTATION)
    RELEASE_ASSERT_WITH_MESSAGE(false, "API unavailable with WPE Backend PlayStation");
#else
    return WebKit::toAPI(WebKit::PlayStationWebView::create(*WebKit::toImpl(configuration)).leakRef());
#endif
}

WKViewRef WKViewCreateWPE(struct wpe_view_backend* backend, WKPageConfigurationRef configuration)
{
#if USE(WPE_BACKEND_PLAYSTATION)
    return WebKit::toAPI(WebKit::PlayStationWebView::create(backend, *WebKit::toImpl(configuration)).leakRef());
#else
    RELEASE_ASSERT_WITH_MESSAGE(false, "API unavailable without WPE Backend PlayStation");
#endif
}

WKPageRef WKViewGetPage(WKViewRef view)
{
    return WebKit::toAPI(WebKit::toImpl(view)->page());
}

void WKViewSetSize(WKViewRef view, WKSize viewSize)
{
    WebKit::toImpl(view)->setViewSize(WebKit::toIntSize(viewSize));
}

static void setViewActivityStateFlag(WKViewRef view, WebCore::ActivityState flag, bool set)
{
    auto viewState = WebKit::toImpl(view)->viewState();
    if (set)
        viewState.add(flag);
    else
        viewState.remove(flag);
    WebKit::toImpl(view)->setViewState(viewState);
}

void WKViewSetFocus(WKViewRef view, bool focused)
{
    setViewActivityStateFlag(view, WebCore::ActivityState::IsFocused, focused);
}

void WKViewSetActive(WKViewRef view, bool active)
{
    setViewActivityStateFlag(view, WebCore::ActivityState::WindowIsActive, active);
}

void WKViewSetVisible(WKViewRef view, bool visible)
{
    setViewActivityStateFlag(view, WebCore::ActivityState::IsVisible, visible);
}

void WKViewWillEnterFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    // FIXME: Replace this and WKViewSetViewClient's enterFullScreen with a listener object.
    WebKit::toImpl(view)->willEnterFullScreen([] (bool) { });
#endif
}

void WKViewDidEnterFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    WebKit::toImpl(view)->didEnterFullScreen();
#endif
}

void WKViewWillExitFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    WebKit::toImpl(view)->willExitFullScreen();
#endif
}

void WKViewDidExitFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    WebKit::toImpl(view)->didExitFullScreen();
#endif
}

void WKViewRequestExitFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    WebKit::toImpl(view)->requestExitFullScreen();
#endif
}

bool WKViewIsFullScreen(WKViewRef view)
{
#if ENABLE(FULLSCREEN_API)
    return WebKit::toImpl(view)->isFullScreen();
#else
    return false;
#endif
}

void WKViewSetViewClient(WKViewRef view, const WKViewClientBase* client)
{
    class ViewClient final : public API::Client<WKViewClientBase>, public API::ViewClient {
    public:
        explicit ViewClient(const WKViewClientBase* client)
        {
            initialize(client);
        }

    private:
        void setViewNeedsDisplay(WebKit::PlayStationWebView& view, const WebCore::Region& region) final
        {
            if (!m_client.setViewNeedsDisplay)
                return;
            m_client.setViewNeedsDisplay(WebKit::toAPI(&view), WebKit::toAPI(region.bounds()), m_client.base.clientInfo);
        }

        void enterFullScreen(WebKit::PlayStationWebView& view, CompletionHandler<void(bool)>&& completionHandler)
        {
            if (!m_client.enterFullScreen)
                return completionHandler(false);
            m_client.enterFullScreen(WebKit::toAPI(&view), m_client.base.clientInfo);

            // FIXME: Replace this and WKViewWillEnterFullScreen with a listener object.
            completionHandler(false);
        }
        
        void exitFullScreen(WebKit::PlayStationWebView& view)
        {
            if (!m_client.exitFullScreen)
                return;
            m_client.exitFullScreen(WebKit::toAPI(&view), m_client.base.clientInfo);
        }
        
        void closeFullScreen(WebKit::PlayStationWebView& view)
        {
            if (!m_client.closeFullScreen)
                return;
            m_client.closeFullScreen(WebKit::toAPI(&view), m_client.base.clientInfo);
        }
        
        void beganEnterFullScreen(WebKit::PlayStationWebView& view, const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
        {
            if (!m_client.beganEnterFullScreen)
                return;
            m_client.beganEnterFullScreen(WebKit::toAPI(&view), WebKit::toAPI(initialFrame), WebKit::toAPI(finalFrame), m_client.base.clientInfo);
        }
        
        void beganExitFullScreen(WebKit::PlayStationWebView& view, const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
        {
            if (!m_client.beganExitFullScreen)
                return;
            m_client.beganExitFullScreen(WebKit::toAPI(&view), WebKit::toAPI(initialFrame), WebKit::toAPI(finalFrame), m_client.base.clientInfo);
        }

        void setCursor(WebKit::PlayStationWebView& view, const WebCore::Cursor& cursor) final
        {
            if (!m_client.setCursor)
                return;
            m_client.setCursor(WebKit::toAPI(&view), toWKCursorType(cursor), m_client.base.clientInfo);
        }
    };

    WebKit::toImpl(view)->setClient(makeUnique<ViewClient>(client));
}
