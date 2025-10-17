/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#include "DefaultUndoController.h"
#include "PageClient.h"
#include "WebFullScreenManagerProxy.h"
#include "WebPageProxy.h"
#include <WebCore/IntSize.h>

namespace WebCore {
enum class DOMPasteAccessCategory : uint8_t;
enum class DOMPasteAccessResponse : uint8_t;
}

namespace WebKit {

class DrawingAreaProxy;
class WebPageNamespace;
class WebView;

enum class ColorControlSupportsAlpha : bool;
enum class UndoOrRedo : bool;

class PageClientImpl : public PageClient
#if ENABLE(FULLSCREEN_API)
    , public WebFullScreenManagerProxyClient
#endif
{
    WTF_MAKE_FAST_ALLOCATED;
#if ENABLE(FULLSCREEN_API)
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageClientImpl);
#endif
public:
    PageClientImpl(WebView&);

    HWND viewWidget();
private:
    // PageClient
    Ref<DrawingAreaProxy> createDrawingAreaProxy(WebProcessProxy&) override;
    void setViewNeedsDisplay(const WebCore::Region&) override;
    void requestScroll(const WebCore::FloatPoint& scrollPosition, const WebCore::IntPoint& scrollOrigin, WebCore::ScrollIsAnimated) override;
    WebCore::FloatPoint viewScrollPosition() override;
    WebCore::IntSize viewSize() override;
    bool isViewWindowActive() override;
    bool isViewFocused() override;
    bool isViewVisible() override;
    bool isViewInWindow() override;
    void processDidExit() override;
    void didRelaunchProcess() override;
    void pageClosed() override;
    void preferencesDidChange() override;
    void toolTipChanged(const WTF::String&, const WTF::String&) override;
    void setCursor(const WebCore::Cursor&) override;
    void setCursorHiddenUntilMouseMoves(bool) override;
    void registerEditCommand(Ref<WebEditCommandProxy>&&, UndoOrRedo) override;
    void clearAllEditCommands() override;
    bool canUndoRedo(UndoOrRedo) override;
    void executeUndoRedo(UndoOrRedo) override;
    WebCore::FloatRect convertToDeviceSpace(const WebCore::FloatRect&) override;
    WebCore::FloatRect convertToUserSpace(const WebCore::FloatRect&) override;
    WebCore::IntPoint screenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntPoint rootViewToScreen(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToScreen(const WebCore::IntRect&) override;
    WebCore::IntPoint accessibilityScreenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToAccessibilityScreen(const WebCore::IntRect&) override;
    void doneWithKeyEvent(const NativeWebKeyboardEvent&, bool wasEventHandled) override;
    RefPtr<WebPopupMenuProxy> createPopupMenuProxy(WebPageProxy&) override;
#if ENABLE(CONTEXT_MENUS)
    Ref<WebContextMenuProxy> createContextMenuProxy(WebPageProxy&, ContextMenuContextData&&, const UserData&) override;
#endif

    RefPtr<WebColorPicker> createColorPicker(WebPageProxy&, const WebCore::Color& intialColor, const WebCore::IntRect&, ColorControlSupportsAlpha, Vector<WebCore::Color>&&) override { return nullptr; }
    RefPtr<WebDataListSuggestionsDropdown> createDataListSuggestionsDropdown(WebPageProxy&) override { return nullptr; }
    RefPtr<WebDateTimePicker> createDateTimePicker(WebPageProxy&) override { return nullptr; }

    void enterAcceleratedCompositingMode(const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode() override;
    void updateAcceleratedCompositingMode(const LayerTreeContext&) override;

    bool handleRunOpenPanel(WebPageProxy*, WebFrameProxy*, const FrameInfoData&, API::OpenPanelParameters*, WebOpenPanelResultListenerProxy*) override;

    void didChangeContentSize(const WebCore::IntSize&) override;
    void didCommitLoadForMainFrame(const String& mimeType, bool useCustomContentProvider) override;
    void didFirstVisuallyNonEmptyLayoutForMainFrame() override;
    void didFinishNavigation(API::Navigation*) override;
    void didFailNavigation(API::Navigation*) override { }
    void didSameDocumentNavigationForMainFrame(SameDocumentNavigationType) override;

#if USE(GRAPHICS_LAYER_WC)
    bool usesOffscreenRendering() const override;
#endif

    // Auxiliary Client Creation
#if ENABLE(FULLSCREEN_API)
    WebFullScreenManagerProxyClient& fullScreenManagerProxyClient() final;
    void setFullScreenClientForTesting(std::unique_ptr<WebFullScreenManagerProxyClient>&&) final;
#endif

#if ENABLE(FULLSCREEN_API)
    // WebFullScreenManagerProxyClient
    void closeFullScreenManager() override;
    bool isFullScreen() override;
    void enterFullScreen(CompletionHandler<void(bool)>&&) override;
    void exitFullScreen() override;
    void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
    void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
#endif

    void didFinishLoadingDataForCustomContentProvider(const String& suggestedFilename, std::span<const uint8_t>) override;

    void navigationGestureDidBegin() override;
    void navigationGestureWillEnd(bool, WebBackForwardListItem&) override;
    void navigationGestureDidEnd(bool, WebBackForwardListItem&) override;
    void navigationGestureDidEnd() override;
    void willRecordNavigationSnapshot(WebBackForwardListItem&) override;
    void didRemoveNavigationGestureSnapshot() override;


#if ENABLE(TOUCH_EVENTS)
    void doneWithTouchEvent(const NativeWebTouchEvent&, bool wasEventHandled) override;
#endif

    void wheelEventWasNotHandledByWebCore(const NativeWebWheelEvent&) override;

    void didChangeBackgroundColor() override;
    void isPlayingAudioWillChange() override;
    void isPlayingAudioDidChange() override;

    void refView() override;
    void derefView() override;

    void didRestoreScrollPosition() override { }

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() override { return WebCore::UserInterfaceLayoutDirection::LTR; }

    void requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect&, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&&) final;

    // Members of PageClientImpl class
    DefaultUndoController m_undoController;

    WebView& m_view;
};

} // namespace WebKit
