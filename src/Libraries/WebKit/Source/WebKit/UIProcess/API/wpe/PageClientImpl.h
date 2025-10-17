/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "WebFullScreenManagerProxy.h"
#include <wtf/TZoneMalloc.h>

struct wpe_view_backend;
typedef struct _AtkObject AtkObject;
typedef struct _WPEView WPEView;

namespace WKWPE {
class View;
}

namespace WebCore {
enum class DOMPasteAccessCategory : uint8_t;
enum class DOMPasteAccessResponse : uint8_t;
}

namespace WebKit {

class WebColorPicker;

struct InputMethodState;
struct UserMessage;

enum class ColorControlSupportsAlpha : bool;
enum class UndoOrRedo : bool;

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
    PageClientImpl(WKWPE::View&);
    virtual ~PageClientImpl();

    struct wpe_view_backend* viewBackend();
#if ENABLE(WPE_PLATFORM)
    WPEView* wpeView() const;
#endif

#if USE(ATK)
    AtkObject* accessible();
#endif

    void sendMessageToWebView(UserMessage&&, CompletionHandler<void(UserMessage&&)>&&);
    void setInputMethodState(std::optional<InputMethodState>&&);
    void callAfterNextPresentationUpdate(CompletionHandler<void()>&&);

private:
    // PageClient
    Ref<DrawingAreaProxy> createDrawingAreaProxy(WebProcessProxy&) override;
    void setViewNeedsDisplay(const WebCore::Region&) override;
    void requestScroll(const WebCore::FloatPoint&, const WebCore::IntPoint&, WebCore::ScrollIsAnimated) override;
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
    void toolTipChanged(const String&, const String&) override;

    void didCommitLoadForMainFrame(const String&, bool) override;

    void didChangeContentSize(const WebCore::IntSize&) override;

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

    void doneWithKeyEvent(const NativeWebKeyboardEvent&, bool) override;
#if ENABLE(TOUCH_EVENTS)
    void doneWithTouchEvent(const NativeWebTouchEvent&, bool) override;
#endif
    void wheelEventWasNotHandledByWebCore(const NativeWebWheelEvent&) override;

    RefPtr<WebPopupMenuProxy> createPopupMenuProxy(WebPageProxy&) override;
#if ENABLE(CONTEXT_MENUS)
    Ref<WebContextMenuProxy> createContextMenuProxy(WebPageProxy&, ContextMenuContextData&&, const UserData&) override;
#endif

    RefPtr<WebColorPicker> createColorPicker(WebPageProxy&, const WebCore::Color& intialColor, const WebCore::IntRect&, ColorControlSupportsAlpha, Vector<WebCore::Color>&&) override;

    RefPtr<WebDataListSuggestionsDropdown> createDataListSuggestionsDropdown(WebPageProxy&) override;

    RefPtr<WebDateTimePicker> createDateTimePicker(WebPageProxy&) override;

    void enterAcceleratedCompositingMode(const LayerTreeContext&) override;
    void exitAcceleratedCompositingMode() override;
    void updateAcceleratedCompositingMode(const LayerTreeContext&) override;

    void didFinishLoadingDataForCustomContentProvider(const String&, std::span<const uint8_t>) override;

    void navigationGestureDidBegin() override;
    void navigationGestureWillEnd(bool, WebBackForwardListItem&) override;
    void navigationGestureDidEnd(bool, WebBackForwardListItem&) override;
    void navigationGestureDidEnd() override;
    void willRecordNavigationSnapshot(WebBackForwardListItem&) override;
    void didRemoveNavigationGestureSnapshot() override;

    void didStartProvisionalLoadForMainFrame() override;
    void didFirstVisuallyNonEmptyLayoutForMainFrame() override;
    void didFinishNavigation(API::Navigation*) override;
    void didFailNavigation(API::Navigation*) override;
    void didSameDocumentNavigationForMainFrame(SameDocumentNavigationType) override;

    void didChangeBackgroundColor() override;
    void isPlayingAudioWillChange() final { }
    void isPlayingAudioDidChange() final { }

    void refView() override;
    void derefView() override;

    void didRestoreScrollPosition() override;

#if ENABLE(FULLSCREEN_API)
    WebFullScreenManagerProxyClient& fullScreenManagerProxyClient() final;
    void setFullScreenClientForTesting(std::unique_ptr<WebFullScreenManagerProxyClient>&&) final;

    void closeFullScreenManager() override;
    bool isFullScreen() override;
    void enterFullScreen(CompletionHandler<void(bool)>&&) override;
    void exitFullScreen() override;
    void beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
    void beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame) override;
#endif

    UnixFileDescriptor hostFileDescriptor() final;
    void requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect&, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&&) final;

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() override;

    void didChangeWebPageID() const override;

    void selectionDidChange() override;

    WebKitWebResourceLoadManager* webResourceLoadManager() override;

    WKWPE::View& m_view;
};

} // namespace WebKit
