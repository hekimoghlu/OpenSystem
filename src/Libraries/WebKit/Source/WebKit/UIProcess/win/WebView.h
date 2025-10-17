/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#include "PageClientImpl.h"
#include "WKView.h"
#include "WebPageProxy.h"
#include <WebCore/COMPtr.h>
#include <WebCore/WindowMessageListener.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class IntSize;
}

namespace WebKit {

class DrawingAreaProxy;

class WebView
    : public API::ObjectImpl<API::Object::Type::View>
    , WebCore::WindowMessageListener {
public:
    static Ref<WebView> create(RECT rect, const API::PageConfiguration& configuration, HWND parentWindow)
    {
        auto webView = adoptRef(*new WebView(rect, configuration, parentWindow));
        webView->initialize();
        return webView;
    }
    ~WebView();

    HWND window() const { return m_window; }
    void setParentWindow(HWND);
    void windowAncestryDidChange();
    WebCore::IntSize viewSize() { return m_viewSize; }
    void setIsInWindow(bool);
    void setIsVisible(bool);
    bool isWindowActive();
    bool isFocused();
    bool isVisible();
    bool isInWindow();
    void setCursor(const WebCore::Cursor&);
    void setOverrideCursor(HCURSOR);
    void setScrollOffsetOnNextResize(const WebCore::IntSize&);
    void initialize();
    void setToolTip(const String&);
    void setUsesOffscreenRendering(bool);
    bool usesOffscreenRendering() const;

    void setViewNeedsDisplay(const WebCore::Region&);

    WebPageProxy* page() const { return m_page.get(); }

    DrawingAreaProxy* drawingArea() { return page() ? page()->drawingArea() : nullptr; }

    void close();

private:
    WebView(RECT, const API::PageConfiguration&, HWND parentWindow);

    static bool registerWebViewWindowClass();
    static LRESULT CALLBACK WebViewWndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

    LRESULT onMouseEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onWheelEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onHorizontalScroll(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onVerticalScroll(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onKeyEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onPaintEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onPrintClientEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onSizeEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onWindowPositionChangedEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onSetFocusEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onKillFocusEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onTimerEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onShowWindowEvent(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onSetCursor(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onMenuCommand(HWND hWnd, UINT message, WPARAM, LPARAM, bool& handled);

    void paint(HDC, const WebCore::IntRect& dirtyRect);
    void setWasActivatedByMouseEvent(bool flag) { m_wasActivatedByMouseEvent = flag; }

    void updateActiveState();
    void updateActiveStateSoon();

    void initializeToolTipWindow();

    void startTrackingMouseLeave();
    void stopTrackingMouseLeave();

    bool shouldInitializeTrackPointHack();

    void closeInternal();

    HCURSOR cursorToShow() const;
    void updateNativeCursor();

    void updateChildWindowGeometries();

    void didCommitLoadForMainFrame(bool useCustomRepresentation);
    void didFinishLoadingDataForCustomRepresentation(const String& suggestedFilename, std::span<const uint8_t>);
    virtual double customRepresentationZoomFactor();
    virtual void setCustomRepresentationZoomFactor(double);

    virtual void findStringInCustomRepresentation(const String&, FindOptions, unsigned maxMatchCount);
    virtual void countStringMatchesInCustomRepresentation(const String&, FindOptions, unsigned maxMatchCount);

    virtual HWND nativeWindow();

    // WebCore::WindowMessageListener
    virtual void windowReceivedMessage(HWND, UINT message, WPARAM, LPARAM);

    HWND m_window { nullptr };
    HWND m_topLevelParentWindow { nullptr };
    HWND m_toolTipWindow { nullptr };
    WTF::String m_toolTip;

    WebCore::IntSize m_nextResizeScrollOffset;

    HCURSOR m_lastCursorSet { nullptr };
    HCURSOR m_webCoreCursor { nullptr };
    HCURSOR m_overrideCursor { nullptr };

    bool m_isInWindow { false };
    bool m_isVisible { false };
    bool m_wasActivatedByMouseEvent { false };
    bool m_trackingMouseLeave { false };
    bool m_isBeingDestroyed { false };
    bool m_usesOffscreenRendering { false };

    std::unique_ptr<WebKit::PageClientImpl> m_pageClient;
    RefPtr<WebPageProxy> m_page;
    WebCore::IntSize m_viewSize;
};

} // namespace WebKit
