/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#include "PageClient.h"
#include <wtf/TZoneMalloc.h>
#if ENABLE(FULLSCREEN_API)
#include "WebFullScreenManagerProxy.h"
#endif

namespace WebKit {

class DrawingAreaProxy;
class PlayStationWebView;

enum class ColorControlSupportsAlpha : bool;

class PageClientImpl final : public PageClient
#if ENABLE(FULLSCREEN_API)
    , public WebFullScreenManagerProxyClient
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(PageClientImpl);
#if ENABLE(FULLSCREEN_API)
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageClientImpl);
#endif
public:
    PageClientImpl(PlayStationWebView&);

#if USE(GRAPHICS_LAYER_WC)
    uint64_t viewWidget();
#endif

private:
    // Create a new drawing area proxy for the given page.
    Ref<DrawingAreaProxy> createDrawingAreaProxy(WebProcessProxy&) override;

    // Tell the view to invalidate the given region. The region is in view coordinates.
    void setViewNeedsDisplay(const WebCore::Region&) override;

    // Tell the view to scroll to the given position, and whether this was a programmatic scroll.
    void requestScroll(const WebCore::FloatPoint& scrollPosition, const WebCore::IntPoint& scrollOrigin, WebCore::ScrollIsAnimated) override;

    // Return the current scroll position (not necessarily the same as the WebCore scroll position, because of scaling, insets etc.)
    WebCore::FloatPoint viewScrollPosition() override;

    // Return the size of the view the page is associated with.
    WebCore::IntSize viewSize() override;

    // Return whether the view's containing window is active.
    bool isViewWindowActive() override;

    // Return whether the view is focused.
    bool isViewFocused() override;

    // Return whether the view is visible.
    bool isViewVisible() override;

    // Return whether the view is in a window.
    bool isViewInWindow() override;

    void processDidExit() override;
    void didRelaunchProcess() override;
    void pageClosed() override;

    void preferencesDidChange() override;

    void toolTipChanged(const String&, const String&) override;

    void didCommitLoadForMainFrame(const String& mimeType, bool useCustomContentProvider) override;

    void didChangeContentSize(const WebCore::IntSize&) override;

    void setCursor(const WebCore::Cursor&) override;
    void setCursorHiddenUntilMouseMoves(bool) override;

    void registerEditCommand(Ref<WebEditCommandProxy>&&, UndoOrRedo) override;
    void clearAllEditCommands() override;
    bool canUndoRedo(UndoOrRedo) override;
    void executeUndoRedo(UndoOrRedo) override;
    void wheelEventWasNotHandledByWebCore(const NativeWebWheelEvent&) override;

    WebCore::FloatRect convertToDeviceSpace(const WebCore::FloatRect&) override;
    WebCore::FloatRect convertToUserSpace(const WebCore::FloatRect&) override;
    WebCore::IntPoint screenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntPoint rootViewToScreen(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToScreen(const WebCore::IntRect&) override;
    WebCore::IntPoint accessibilityScreenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToAccessibilityScreen(const WebCore::IntRect&) override;

    void doneWithKeyEvent(const NativeWebKeyboardEvent&, bool wasEventHandled) override;
#if ENABLE(TOUCH_EVENTS)
    void doneWithTouchEvent(const NativeWebTouchEvent&, bool) override;
#endif

    RefPtr<WebPopupMenuProxy> createPopupMenuProxy(WebPageProxy&) override;

    RefPtr<WebColorPicker> createColorPicker(WebPageProxy&, const WebCore::Color& intialColor, const WebCore::IntRect&, ColorControlSupportsAlpha, Vector<WebCore::Color>&&) override { return nullptr; }
    RefPtr<WebDataListSuggestionsDropdown> createDataListSuggestionsDropdown(WebPageProxy&) override { return nullptr; }
    RefPtr<WebDateTimePicker> createDateTimePicker(WebPageProxy&) override { return nullptr; }

    void enterAcceleratedCompositingMode(const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode() override;
    void updateAcceleratedCompositingMode(const LayerTreeContext&) override;

#if USE(GRAPHICS_LAYER_WC)
    bool usesOffscreenRendering() const override;
#endif

    // Auxiliary Client Creation
#if ENABLE(FULLSCREEN_API)
    WebFullScreenManagerProxyClient& fullScreenManagerProxyClient() override;
    void setFullScreenClientForTesting(std::unique_ptr<WebFullScreenManagerProxyClient>&&) override;

    void closeFullScreenManager() override;
    bool isFullScreen() override;
    void enterFullScreen(CompletionHandler<void(bool)>&&) override;
    void exitFullScreen() override;
    void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
    void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
#endif

    // Custom representations.
    void didFinishLoadingDataForCustomContentProvider(const String& suggestedFilename, std::span<const uint8_t>) override;

    void navigationGestureDidBegin() override;
    void navigationGestureWillEnd(bool willNavigate, WebBackForwardListItem&) override;
    void navigationGestureDidEnd(bool willNavigate, WebBackForwardListItem&) override;
    void navigationGestureDidEnd() override;
    void willRecordNavigationSnapshot(WebBackForwardListItem&) override;
    void didRemoveNavigationGestureSnapshot() override;

    void didFirstVisuallyNonEmptyLayoutForMainFrame() override;
    void didFinishNavigation(API::Navigation*) override;
    void didFailNavigation(API::Navigation*) override;
    void didSameDocumentNavigationForMainFrame(SameDocumentNavigationType) override;

    void didChangeBackgroundColor() override;
    void isPlayingAudioWillChange() override;
    void isPlayingAudioDidChange() override;

    void refView() override;
    void derefView() override;

    void didRestoreScrollPosition() override;

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() override;

    void requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect&, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&&) override;

#if USE(WPE_RENDERER)
    UnixFileDescriptor hostFileDescriptor() override;
#endif

    PlayStationWebView& m_view;
};

} // namespace WebKit
