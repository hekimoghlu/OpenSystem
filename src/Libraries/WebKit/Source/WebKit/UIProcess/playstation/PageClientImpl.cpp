/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#include "PageClientImpl.h"

#include "DrawingAreaProxyCoordinatedGraphics.h"
#include "PlayStationWebView.h"
#include "WebPageProxy.h"
#include <WebCore/DOMPasteAccess.h>
#include <WebCore/NotImplemented.h>
#include <wtf/TZoneMallocInlines.h>

#if USE(GRAPHICS_LAYER_WC)
#include "DrawingAreaProxyWC.h"
#endif

#if USE(WPE_RENDERER)
#include <wpe/wpe.h>
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageClientImpl);

PageClientImpl::PageClientImpl(PlayStationWebView& view)
    : m_view(view)
{
}

#if USE(GRAPHICS_LAYER_WC) && USE(WPE_RENDERER)
uint64_t PageClientImpl::viewWidget()
{
    return 0;
}
#endif

// PageClient's pure virtual functions
Ref<DrawingAreaProxy> PageClientImpl::createDrawingAreaProxy(WebProcessProxy& webProcessProxy)
{
#if USE(GRAPHICS_LAYER_WC)
    return DrawingAreaProxyWC::create(*m_view.page(), webProcessProxy);
#else
    return DrawingAreaProxyCoordinatedGraphics::create(*m_view.page(), webProcessProxy);
#endif
}

void PageClientImpl::setViewNeedsDisplay(const WebCore::Region& region)
{
    m_view.setViewNeedsDisplay(region);
}

void PageClientImpl::requestScroll(const WebCore::FloatPoint& scrollPosition, const WebCore::IntPoint& scrollOrigin, WebCore::ScrollIsAnimated)
{
}

WebCore::FloatPoint PageClientImpl::viewScrollPosition()
{
    return WebCore::FloatPoint { };
}

WebCore::IntSize PageClientImpl::viewSize()
{
    return m_view.viewSize();
}

bool PageClientImpl::isViewWindowActive()
{
    return m_view.viewState().contains(WebCore::ActivityState::WindowIsActive);
}

bool PageClientImpl::isViewFocused()
{
    return m_view.viewState().contains(WebCore::ActivityState::IsFocused);
}

bool PageClientImpl::isViewVisible()
{
    return m_view.viewState().contains(WebCore::ActivityState::IsVisible);
}

bool PageClientImpl::isViewInWindow()
{
    return m_view.viewState().contains(WebCore::ActivityState::IsInWindow);
}

void PageClientImpl::processDidExit()
{
}

void PageClientImpl::didRelaunchProcess()
{
}

void PageClientImpl::pageClosed()
{
}

void PageClientImpl::preferencesDidChange()
{
}

void PageClientImpl::toolTipChanged(const String&, const String&)
{
}

void PageClientImpl::didCommitLoadForMainFrame(const String& mimeType, bool useCustomContentProvider)
{
    notImplemented();
}

void PageClientImpl::didChangeContentSize(const WebCore::IntSize& size)
{
    notImplemented();
}

void PageClientImpl::setCursor(const WebCore::Cursor& cursor)
{
    m_view.setCursor(cursor);
}

void PageClientImpl::setCursorHiddenUntilMouseMoves(bool)
{
}

void PageClientImpl::registerEditCommand(Ref<WebEditCommandProxy>&&, UndoOrRedo)
{
}

void PageClientImpl::clearAllEditCommands()
{
}

bool PageClientImpl::canUndoRedo(UndoOrRedo)
{
    return false;
}

void PageClientImpl::executeUndoRedo(UndoOrRedo)
{
}

void PageClientImpl::wheelEventWasNotHandledByWebCore(const NativeWebWheelEvent&)
{
}

WebCore::FloatRect PageClientImpl::convertToDeviceSpace(const WebCore::FloatRect& rect)
{
    return rect;
}

WebCore::FloatRect PageClientImpl::convertToUserSpace(const WebCore::FloatRect& rect)
{
    return rect;
}

WebCore::IntPoint PageClientImpl::screenToRootView(const WebCore::IntPoint& point)
{
    return point;
}

WebCore::IntPoint PageClientImpl::rootViewToScreen(const WebCore::IntPoint& point)
{
    return point;
}

WebCore::IntRect PageClientImpl::rootViewToScreen(const WebCore::IntRect& rect)
{
    return rect;
}

WebCore::IntPoint PageClientImpl::accessibilityScreenToRootView(const WebCore::IntPoint& point)
{
    return screenToRootView(point);
}

WebCore::IntRect PageClientImpl::rootViewToAccessibilityScreen(const WebCore::IntRect& rect)
{
    return rootViewToScreen(rect);    
}

void PageClientImpl::doneWithKeyEvent(const NativeWebKeyboardEvent& event, bool wasEventHandled)
{
    notImplemented();
}

#if ENABLE(TOUCH_EVENTS)
void PageClientImpl::doneWithTouchEvent(const NativeWebTouchEvent& event, bool wasEventHandled)
{
    notImplemented();
}
#endif // ENABLE(TOUCH_EVENTS)

RefPtr<WebPopupMenuProxy> PageClientImpl::createPopupMenuProxy(WebPageProxy&  pageProxy)
{
    notImplemented();
    return { };
}

#if USE(GRAPHICS_LAYER_WC)
bool PageClientImpl::usesOffscreenRendering() const
{
    return false;
}
#endif

void PageClientImpl::enterAcceleratedCompositingMode(const LayerTreeContext& context)
{
    notImplemented();
}

void PageClientImpl::exitAcceleratedCompositingMode()
{
    notImplemented();
}

void PageClientImpl::updateAcceleratedCompositingMode(const LayerTreeContext&)
{
    notImplemented();
}

#if ENABLE(FULLSCREEN_API)
WebFullScreenManagerProxyClient& PageClientImpl::fullScreenManagerProxyClient()
{
    return *(WebFullScreenManagerProxyClient*)this;
}

void PageClientImpl::setFullScreenClientForTesting(std::unique_ptr<WebFullScreenManagerProxyClient>&&)
{
}

void PageClientImpl::closeFullScreenManager()
{
    m_view.closeFullScreenManager();
}

bool PageClientImpl::isFullScreen()
{
    return m_view.isFullScreen();
}

void PageClientImpl::enterFullScreen(CompletionHandler<void(bool)>&& completionHandler)
{
    m_view.enterFullScreen(WTFMove(completionHandler));
}

void PageClientImpl::exitFullScreen()
{
    m_view.exitFullScreen();
}

void PageClientImpl::beganEnterFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
{
    m_view.beganEnterFullScreen(initialFrame, finalFrame);
}

void PageClientImpl::beganExitFullScreen(const WebCore::IntRect& initialFrame, const WebCore::IntRect& finalFrame)
{
    m_view.beganExitFullScreen(initialFrame, finalFrame);
}
#endif

// Custom representations.
void PageClientImpl::didFinishLoadingDataForCustomContentProvider(const String& suggestedFilename, std::span<const uint8_t>)
{
}

void PageClientImpl::navigationGestureDidBegin()
{
}

void PageClientImpl::navigationGestureWillEnd(bool willNavigate, WebBackForwardListItem&)
{
}

void PageClientImpl::navigationGestureDidEnd(bool willNavigate, WebBackForwardListItem&)
{
}

void PageClientImpl::navigationGestureDidEnd()
{
}

void PageClientImpl::willRecordNavigationSnapshot(WebBackForwardListItem&)
{
}

void PageClientImpl::didRemoveNavigationGestureSnapshot()
{
}

void PageClientImpl::didFirstVisuallyNonEmptyLayoutForMainFrame()
{
}

void PageClientImpl::didFinishNavigation(API::Navigation*)
{
}

void PageClientImpl::didFailNavigation(API::Navigation*)
{
}

void PageClientImpl::didSameDocumentNavigationForMainFrame(SameDocumentNavigationType)
{
}

void PageClientImpl::didChangeBackgroundColor()
{
}

void PageClientImpl::isPlayingAudioWillChange()
{
}

void PageClientImpl::isPlayingAudioDidChange()
{
}

void PageClientImpl::refView()
{
}

void PageClientImpl::derefView()
{
}

void PageClientImpl::didRestoreScrollPosition()
{
}

WebCore::UserInterfaceLayoutDirection PageClientImpl::userInterfaceLayoutDirection()
{
    return WebCore::UserInterfaceLayoutDirection::LTR;
}

void PageClientImpl::requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::DOMPasteRequiresInteraction, const WebCore::IntRect&, const String&, CompletionHandler<void(WebCore::DOMPasteAccessResponse)>&& completionHandler)
{
    completionHandler(WebCore::DOMPasteAccessResponse::DeniedForGesture);
}

#if USE(WPE_RENDERER)
UnixFileDescriptor PageClientImpl::hostFileDescriptor()
{
    return UnixFileDescriptor { wpe_view_backend_get_renderer_host_fd(m_view.backend()), UnixFileDescriptor::Adopt };
}
#endif

} // namespace WebKit
