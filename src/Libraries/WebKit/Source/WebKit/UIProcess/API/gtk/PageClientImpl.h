/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#include <gtk/gtk.h>
#include <memory>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
enum class DOMPasteAccessPolicy : uint8_t;
}

namespace WebKit {

class DrawingAreaProxy;
class WebPageNamespace;

enum class ColorControlSupportsAlpha : bool;
enum class UndoOrRedo : bool;

class PageClientImpl : public PageClient
#if ENABLE(FULLSCREEN_API)
    , public WebFullScreenManagerProxyClient
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(PageClientImpl);
#if ENABLE(FULLSCREEN_API)
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageClientImpl);
#endif
public:
    explicit PageClientImpl(GtkWidget*);

    GtkWidget* viewWidget() { return m_viewWidget; }

private:
    // PageClient
    Ref<DrawingAreaProxy> createDrawingAreaProxy(WebProcessProxy&) override;
    void setViewNeedsDisplay(const WebCore::Region&) override;
    void requestScroll(const WebCore::FloatPoint& scrollPosition, const WebCore::IntPoint& scrollOrigin, WebCore::ScrollIsAnimated) override;
    void requestScrollToRect(const WebCore::FloatRect& targetRect, const WebCore::FloatPoint& origin) override;
    WebCore::FloatPoint viewScrollPosition() override;
    WebCore::IntSize viewSize() override;
    bool isViewWindowActive() override;
    bool isViewFocused() override;
    bool isViewVisible() override;
    bool isViewInWindow() override;
    void processWillSwap() override;
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
    WebCore::IntRect rootViewToScreen(const WebCore::IntRect&) override;
    WebCore::IntPoint rootViewToScreen(const WebCore::IntPoint&) override;
    WebCore::IntPoint accessibilityScreenToRootView(const WebCore::IntPoint&) override;
    WebCore::IntRect rootViewToAccessibilityScreen(const WebCore::IntRect&) override;
    void doneWithKeyEvent(const NativeWebKeyboardEvent&, bool wasEventHandled) override;
    RefPtr<WebPopupMenuProxy> createPopupMenuProxy(WebPageProxy&) override;
#if ENABLE(CONTEXT_MENUS)
    Ref<WebContextMenuProxy> createContextMenuProxy(WebPageProxy&, ContextMenuContextData&&, const UserData&) override;
#endif // ENABLE(CONTEXT_MENUS)
    RefPtr<WebColorPicker> createColorPicker(WebPageProxy&, const WebCore::Color& initialColor, const WebCore::IntRect&, ColorControlSupportsAlpha, Vector<WebCore::Color>&&) override;
    RefPtr<WebDateTimePicker> createDateTimePicker(WebPageProxy&) override;
    RefPtr<WebDataListSuggestionsDropdown> createDataListSuggestionsDropdown(WebPageProxy&) override;
    Ref<WebCore::ValidationBubble> createValidationBubble(const String& message, const WebCore::ValidationBubble::Settings&) final;
    void selectionDidChange() override;
    RefPtr<ViewSnapshot> takeViewSnapshot(std::optional<WebCore::IntRect>&&) override;
#if ENABLE(DRAG_SUPPORT)
    void startDrag(WebCore::SelectionData&&, OptionSet<WebCore::DragOperation>, RefPtr<WebCore::ShareableBitmap>&& dragImage, WebCore::IntPoint&& dragImageHotspot) override;
    void didPerformDragControllerAction() override;
#endif

    void enterAcceleratedCompositingMode(const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode() override;
    void updateAcceleratedCompositingMode(const LayerTreeContext&) override;

    void didChangeContentSize(const WebCore::IntSize&) override;
    void didCommitLoadForMainFrame(const String& mimeType, bool useCustomContentProvider) override;
    void didStartProvisionalLoadForMainFrame() override;
    void didFirstVisuallyNonEmptyLayoutForMainFrame() override;
    void didFinishNavigation(API::Navigation*) override;
    void didFailNavigation(API::Navigation*) override;
    void didSameDocumentNavigationForMainFrame(SameDocumentNavigationType) override;

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

    void refView() override;
    void derefView() override;

    void didRestoreScrollPosition() override;
    void isPlayingAudioWillChange() final { }
    void isPlayingAudioDidChange() final { }

    void requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect&, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&&) final;

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() override;

    bool effectiveAppearanceIsDark() const override;

    void didChangeWebPageID() const override;

    void makeViewBlank(bool) override;

    WebCore::Color accentColor() override;

    WebKitWebResourceLoadManager* webResourceLoadManager() override;

    // Members of PageClientImpl class
    GtkWidget* m_viewWidget;
    DefaultUndoController m_undoController;
};

} // namespace WebKit
