/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#include <WebCore/FloatPoint.h>
#include <WebCore/PageOverlay.h>
#include <WebCore/SimpleRange.h>
#include <wtf/RetainPtr.h>
#include <wtf/WeakPtr.h>

#if HAVE(SECURE_ACTION_CONTEXT)
OBJC_CLASS DDSecureActionContext;
using WKDDActionContext = DDSecureActionContext;
#else
OBJC_CLASS DDActionContext;
using WKDDActionContext = DDActionContext;
#endif

namespace WebCore {
class IntRect;
}

namespace WebKit {

class WebFrame;
class WebPage;

class WebPageOverlay : public API::ObjectImpl<API::Object::Type::BundlePageOverlay>, public CanMakeWeakPtr<WebPageOverlay>, private WebCore::PageOverlayClient {
public:
    struct ActionContext;

    class Client {
    public:
        virtual ~Client() { }

        virtual void willMoveToPage(WebPageOverlay&, WebPage*) = 0;
        virtual void didMoveToPage(WebPageOverlay&, WebPage*) = 0;
        virtual void drawRect(WebPageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect& dirtyRect) = 0;
        virtual bool mouseEvent(WebPageOverlay&, const WebCore::PlatformMouseEvent&) = 0;
        virtual void didScrollFrame(WebPageOverlay&, WebFrame*) { }

#if PLATFORM(MAC)
        virtual std::optional<ActionContext> actionContextForResultAtPoint(WebPageOverlay&, WebCore::FloatPoint) { return std::nullopt; }
        virtual void dataDetectorsDidPresentUI(WebPageOverlay&) { }
        virtual void dataDetectorsDidChangeUI(WebPageOverlay&) { }
        virtual void dataDetectorsDidHideUI(WebPageOverlay&) { }
#endif

        virtual bool copyAccessibilityAttributeStringValueForPoint(WebPageOverlay&, String /* attribute */, WebCore::FloatPoint /* parameter */, String& /* value */) { return false; }
        virtual bool copyAccessibilityAttributeBoolValueForPoint(WebPageOverlay&, String /* attribute */, WebCore::FloatPoint /* parameter */, bool& /* value */) { return false; }
        virtual Vector<String> copyAccessibilityAttributeNames(WebPageOverlay&, bool /* parameterizedNames */) { return Vector<String>(); }
    };

    static Ref<WebPageOverlay> create(std::unique_ptr<Client>, WebCore::PageOverlay::OverlayType = WebCore::PageOverlay::OverlayType::View);
    static WebPageOverlay* fromCoreOverlay(WebCore::PageOverlay&);
    virtual ~WebPageOverlay();

    void setNeedsDisplay(const WebCore::IntRect& dirtyRect);
    void setNeedsDisplay();

    void clear();

    WebCore::PageOverlay* coreOverlay() const { return m_overlay.get(); }
    Client& client() const { return *m_client; }

#if PLATFORM(MAC)
    struct ActionContext {
        RetainPtr<WKDDActionContext> context;
        WebCore::SimpleRange range;
    };
    std::optional<ActionContext> actionContextForResultAtPoint(WebCore::FloatPoint);
    void dataDetectorsDidPresentUI();
    void dataDetectorsDidChangeUI();
    void dataDetectorsDidHideUI();
#endif

private:
    WebPageOverlay(std::unique_ptr<Client>, WebCore::PageOverlay::OverlayType);

    // WebCore::PageOverlayClient
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect& dirtyRect) override;
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) override;
    void didScrollFrame(WebCore::PageOverlay&, WebCore::LocalFrame&) override;

    bool copyAccessibilityAttributeStringValueForPoint(WebCore::PageOverlay&, String /* attribute */, WebCore::FloatPoint /* parameter */, String& value) override;
    bool copyAccessibilityAttributeBoolValueForPoint(WebCore::PageOverlay&, String /* attribute */, WebCore::FloatPoint /* parameter */, bool& value) override;
    Vector<String> copyAccessibilityAttributeNames(WebCore::PageOverlay&, bool /* parameterizedNames */) override;

    RefPtr<WebCore::PageOverlay> m_overlay;
    std::unique_ptr<Client> m_client;
};

} // namespace WebKit
