/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include "InputMethodFilter.h"
#include "KeyAutoRepeatHandler.h"
#include "PageClientImpl.h"
#include "WebFullScreenManagerProxy.h"
#include "WebKitWebViewAccessible.h"
#include "WebPageProxy.h"
#include <WebCore/ActivityState.h>
#include <memory>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>

typedef struct _WebKitInputMethodContext WebKitInputMethodContext;
typedef struct _WPEView WPEView;
struct wpe_view_backend;

namespace API {
class ViewClient;
}

namespace WebCore {
class Cursor;
struct CompositionUnderline;
}

namespace WebKit {
class WebKitWebResourceLoadManager;
struct EditingRange;
struct UserMessage;
}

namespace WKWPE {

class View : public API::ObjectImpl<API::Object::Type::View> {
public:
    virtual ~View();

    // Client methods
    void setClient(std::unique_ptr<API::ViewClient>&&);
    void frameDisplayed();
    void willStartLoad();
    void didChangePageID();
    void didReceiveUserMessage(WebKit::UserMessage&&, CompletionHandler<void(WebKit::UserMessage&&)>&&);
    WebKit::WebKitWebResourceLoadManager* webResourceLoadManager();

    void setInputMethodContext(WebKitInputMethodContext*);
    WebKitInputMethodContext* inputMethodContext() const;
    void setInputMethodState(std::optional<WebKit::InputMethodState>&&);

#if ENABLE(FULLSCREEN_API)
    bool isFullScreen() const;
    void willEnterFullScreen(CompletionHandler<void(bool)>&&);
    void willExitFullScreen();
#endif

    void selectionDidChange();
    void close();

    WebKit::WebPageProxy& page() { return *m_pageProxy; }
    API::ViewClient& client() const { return *m_client; }
    const WebCore::IntSize& size() const { return m_size; }
    OptionSet<WebCore::ActivityState> viewState() const { return m_viewStateFlags; }

#if USE(ATK)
    WebKitWebViewAccessible* accessible() const;
#endif

    virtual struct wpe_view_backend* backend() const { return nullptr; }
#if ENABLE(WPE_PLATFORM)
    virtual WPEView* wpeView() const { return nullptr; }
#endif
#if ENABLE(POINTER_LOCK)
    virtual void requestPointerLock() { };
    virtual void didLosePointerLock() { };
#endif

    virtual void setCursor(const WebCore::Cursor&) { };
    virtual void synthesizeCompositionKeyPress(const String&, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<WebKit::EditingRange>&&) = 0;
    virtual void callAfterNextPresentationUpdate(CompletionHandler<void()>&&) = 0;

protected:
    View();

    void createWebPage(const API::PageConfiguration&);
    void setSize(const WebCore::IntSize&);

    std::unique_ptr<API::ViewClient> m_client;
    std::unique_ptr<WebKit::PageClientImpl> m_pageClient;
    RefPtr<WebKit::WebPageProxy> m_pageProxy;
    WebCore::IntSize m_size;
    OptionSet<WebCore::ActivityState> m_viewStateFlags;
#if ENABLE(FULLSCREEN_API)
    WebKit::WebFullScreenManagerProxy::FullscreenState m_fullscreenState { WebKit::WebFullScreenManagerProxy::FullscreenState::NotInFullscreen };
#endif
    WebKit::InputMethodFilter m_inputMethodFilter;
    WebKit::KeyAutoRepeatHandler m_keyAutoRepeatHandler;
#if USE(ATK)
    mutable GRefPtr<WebKitWebViewAccessible> m_accessible;
#endif
};

} // namespace WKWPE
