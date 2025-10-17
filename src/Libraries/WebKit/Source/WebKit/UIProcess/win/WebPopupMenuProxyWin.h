/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

#include "PlatformPopupMenuData.h"
#include "WebPopupItem.h"
#include "WebPopupMenuProxy.h"
#include <OleAcc.h>
#include <WebCore/ScrollableArea.h>
#include <WebCore/Scrollbar.h>
#include <wtf/TZoneMalloc.h>

#if USE(SKIA)
class SkSurface;
#endif

namespace WebKit {

class WebView;

class WebPopupMenuProxyWin final : public CanMakeCheckedPtr<WebPopupMenuProxyWin>, public WebPopupMenuProxy, private WebCore::ScrollableArea {
    WTF_MAKE_TZONE_ALLOCATED(WebPopupMenuProxyWin);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebPopupMenuProxyWin);
public:
    static Ref<WebPopupMenuProxyWin> create(WebView* webView, WebPopupMenuProxy::Client& client)
    {
        return adoptRef(*new WebPopupMenuProxyWin(webView, client));
    }
    ~WebPopupMenuProxyWin();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void showPopupMenu(const WebCore::IntRect&, WebCore::TextDirection, double pageScaleFactor, const Vector<WebPopupItem>&, const PlatformPopupMenuData&, int32_t selectedIndex) override;
    void hidePopupMenu() override;

    bool setFocusedIndex(int index, bool hotTracking = false);

    void hide() { hidePopupMenu(); }

    String debugDescription() const final;

private:
    WebPopupMenuProxyWin(WebView*, WebPopupMenuProxy::Client&);

    // ScrollableArea
    WebCore::ScrollPosition scrollPosition() const override;
    void setScrollOffset(const WebCore::IntPoint&) override;

    void invalidateScrollbarRect(WebCore::Scrollbar&, const WebCore::IntRect&) override;
    void invalidateScrollCornerRect(const WebCore::IntRect&) override { }
    bool isActive() const override { return true; }
    bool isScrollCornerVisible() const override { return false; }
    WebCore::IntRect scrollCornerRect() const override { return WebCore::IntRect(); }
    WebCore::Scrollbar* verticalScrollbar() const override { return m_scrollbar.get(); }
    WebCore::ScrollableArea* enclosingScrollableArea() const override { return 0; }
    WebCore::IntSize visibleSize() const override;
    WebCore::IntSize contentsSize() const override;
    WebCore::IntRect scrollableAreaBoundingBox(bool* = nullptr) const override;
    bool shouldPlaceVerticalScrollbarOnLeft() const override;
    bool forceUpdateScrollbarsOnMainThreadForPerformanceTesting() const override { return false; }
    bool isScrollableOrRubberbandable() override { return true; }
    bool hasScrollableOrRubberbandableAncestor() override { return true; }

    // NOTE: This should only be called by the overriden setScrollOffset from ScrollableArea.
    void scrollTo(int offset);

    static bool registerWindowClass();
    static LRESULT CALLBACK WebPopupMenuProxyWndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

    // Message pump messages.
    LRESULT onMouseActivate(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onSize(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onKeyDown(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onChar(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onMouseMove(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onLButtonDown(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onLButtonUp(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onMouseWheel(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onPaint(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onPrintClient(HWND, UINT message, WPARAM, LPARAM, bool& handled);
    LRESULT onGetObject(HWND, UINT message, WPARAM, LPARAM, bool &handled);

    void calculatePositionAndSize(const WebCore::IntRect&);
    WebCore::IntRect clientRect() const;
    void invalidateItem(int index);


    int itemHeight() const { return m_itemHeight; }
    const WebCore::IntRect& windowRect() const { return m_windowRect; }
    int wheelDelta() const { return m_wheelDelta; }
    void setWasClicked(bool b = true) { m_wasClicked = b; }
    bool wasClicked() const { return m_wasClicked; }
    bool scrollbarCapturingMouse() const { return m_scrollbarCapturingMouse; }
    void setScrollbarCapturingMouse(bool b) { m_scrollbarCapturingMouse = b; }

    bool up(unsigned lines = 1);
    bool down(unsigned lines = 1);

    void paint(const WebCore::IntRect& damageRect, HDC = 0);
    int visibleItems() const;
    int listIndexAtPoint(const WebCore::IntPoint&) const;
    int focusedIndex() const;
    void focusFirst();
    void focusLast();
    bool scrollToRevealSelection();
    void incrementWheelDelta(int delta);
    void reduceWheelDelta(int delta);

    WebView* m_webView;
    Vector<WebPopupItem> m_items;
    PlatformPopupMenuData m_data;
    int m_newSelectedIndex { 0 };

    RefPtr<WebCore::Scrollbar> m_scrollbar;
#if USE(CAIRO)
    GDIObject<HDC> m_DC;
    GDIObject<HBITMAP> m_bmp;
#elif USE(SKIA)
    sk_sp<SkSurface> m_surface;
#endif
    HWND m_popup { nullptr };
    WebCore::IntRect m_windowRect;
    WebCore::IntSize m_clientSize;

    float m_itemHeight { 0 };
    int m_scrollOffset { 0 };
    int m_wheelDelta { 0 };
    int m_focusedIndex { 0 };
    int m_hoveredIndex { 0 };
    bool m_wasClicked { false };
    bool m_scrollbarCapturingMouse { false };
    bool m_showPopup { false };
    int m_scaleFactor { 1 };
};

} // namespace WebKit
