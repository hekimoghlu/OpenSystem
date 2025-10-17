/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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

#include "APIPageConfiguration.h"
#include "InputMethodState.h"
#include "RendererBufferFormat.h"
#include "SameDocumentNavigationType.h"
#include "ViewGestureController.h"
#include "ViewSnapshotStore.h"
#include "WebContextMenuProxyGtk.h"
#include "WebHitTestResultData.h"
#include "WebInspectorUIProxy.h"
#include "WebKitInputMethodContext.h"
#include "WebKitWebViewBase.h"
#include "WebKitWebViewBaseInternal.h"
#include "WebPageProxy.h"
#include <WebCore/DragActions.h>
#include <WebCore/GRefPtrGtk.h>
#include <WebCore/GUniquePtrGtk.h>
#include <WebCore/SelectionData.h>
#include <WebCore/ShareableBitmap.h>

WebKitWebViewBase* webkitWebViewBaseCreate(const API::PageConfiguration&);
WebKit::WebPageProxy* webkitWebViewBaseGetPage(WebKitWebViewBase*);
double webkitWebViewBaseGetPageScale(WebKitWebViewBase*);
void webkitWebViewBaseCreateWebPage(WebKitWebViewBase*, Ref<API::PageConfiguration>&&);
void webkitWebViewBaseSetTooltipText(WebKitWebViewBase*, const char*);
void webkitWebViewBaseSetTooltipArea(WebKitWebViewBase*, const WebCore::IntRect&);
void webkitWebViewBaseSetMouseIsOverScrollbar(WebKitWebViewBase*, WebKit::WebHitTestResultData::IsScrollbar);
void webkitWebViewBasePropagateKeyEvent(WebKitWebViewBase*, GdkEvent*);
void webkitWebViewBasePropagateWheelEvent(WebKitWebViewBase*, GdkEvent*);
void webkitWebViewBaseChildMoveResize(WebKitWebViewBase*, GtkWidget*, const WebCore::IntRect&);
#if ENABLE(FULLSCREEN_API)
void webkitWebViewBaseWillEnterFullScreen(WebKitWebViewBase*, CompletionHandler<void(bool)>&&);
void webkitWebViewBaseEnterFullScreen(WebKitWebViewBase*);
void webkitWebViewBaseWillExitFullScreen(WebKitWebViewBase*);
void webkitWebViewBaseExitFullScreen(WebKitWebViewBase*);
bool webkitWebViewBaseIsFullScreen(WebKitWebViewBase*);
#endif
void webkitWebViewBaseSetInspectorViewSize(WebKitWebViewBase*, unsigned size);
#if ENABLE(CONTEXT_MENUS)
void webkitWebViewBaseSetActiveContextMenuProxy(WebKitWebViewBase*, WebKit::WebContextMenuProxyGtk*);
WebKit::WebContextMenuProxyGtk* webkitWebViewBaseGetActiveContextMenuProxy(WebKitWebViewBase*);
#endif // ENABLE(CONTEXT_MENUS)

#if USE(GTK4)
GRefPtr<GdkEvent> webkitWebViewBaseTakeContextMenuEvent(WebKitWebViewBase*);
#else
GUniquePtr<GdkEvent> webkitWebViewBaseTakeContextMenuEvent(WebKitWebViewBase*);
#endif

void webkitWebViewBaseSetInputMethodState(WebKitWebViewBase*, std::optional<WebKit::InputMethodState>&&);
void webkitWebViewBaseUpdateTextInputState(WebKitWebViewBase*);
void webkitWebViewBaseSetContentsSize(WebKitWebViewBase*, const WebCore::IntSize&);

void webkitWebViewBaseSetFocus(WebKitWebViewBase*, bool focused);
void webkitWebViewBaseSetEditable(WebKitWebViewBase*, bool editable);
WebCore::IntSize webkitWebViewBaseGetViewSize(WebKitWebViewBase*);
bool webkitWebViewBaseIsInWindowActive(WebKitWebViewBase*);
bool webkitWebViewBaseIsFocused(WebKitWebViewBase*);
bool webkitWebViewBaseIsVisible(WebKitWebViewBase*);
bool webkitWebViewBaseIsInWindow(WebKitWebViewBase*);

void webkitWebViewBaseAddDialog(WebKitWebViewBase*, GtkWidget*);
void webkitWebViewBaseAddWebInspector(WebKitWebViewBase*, GtkWidget* inspector, WebKit::AttachmentSide);
void webkitWebViewBaseRemoveWebInspector(WebKitWebViewBase*, GtkWidget*);
void webkitWebViewBaseResetClickCounter(WebKitWebViewBase*);
void webkitWebViewBaseEnterAcceleratedCompositingMode(WebKitWebViewBase*, const WebKit::LayerTreeContext&);
void webkitWebViewBaseUpdateAcceleratedCompositingMode(WebKitWebViewBase*, const WebKit::LayerTreeContext&);
void webkitWebViewBaseExitAcceleratedCompositingMode(WebKitWebViewBase*);
void webkitWebViewBaseWillSwapWebProcess(WebKitWebViewBase*);
void webkitWebViewBaseDidExitWebProcess(WebKitWebViewBase*);
void webkitWebViewBaseDidRelaunchWebProcess(WebKitWebViewBase*);
void webkitWebViewBasePageClosed(WebKitWebViewBase*);

#if ENABLE(DRAG_SUPPORT)
void webkitWebViewBaseStartDrag(WebKitWebViewBase*, WebCore::SelectionData&&, OptionSet<WebCore::DragOperation>, RefPtr<WebCore::ShareableBitmap>&&, WebCore::IntPoint&& dragImageHotspot);
void webkitWebViewBaseDidPerformDragControllerAction(WebKitWebViewBase*);
#endif

RefPtr<WebKit::ViewSnapshot> webkitWebViewBaseTakeViewSnapshot(WebKitWebViewBase*, std::optional<WebCore::IntRect>&&);
void webkitWebViewBaseSetEnableBackForwardNavigationGesture(WebKitWebViewBase*, bool enabled);
WebKit::ViewGestureController* webkitWebViewBaseViewGestureController(WebKitWebViewBase*);

bool webkitWebViewBaseBeginBackSwipeForTesting(WebKitWebViewBase*);
bool webkitWebViewBaseCompleteBackSwipeForTesting(WebKitWebViewBase*);

GVariant* webkitWebViewBaseContentsOfUserInterfaceItem(WebKitWebViewBase*, const char* userInterfaceItem);

void webkitWebViewBaseDidStartProvisionalLoadForMainFrame(WebKitWebViewBase*);
void webkitWebViewBaseDidFirstVisuallyNonEmptyLayoutForMainFrame(WebKitWebViewBase*);
void webkitWebViewBaseDidFinishNavigation(WebKitWebViewBase*, API::Navigation*);
void webkitWebViewBaseDidFailNavigation(WebKitWebViewBase*, API::Navigation*);
void webkitWebViewBaseDidSameDocumentNavigationForMainFrame(WebKitWebViewBase*, WebKit::SameDocumentNavigationType);
void webkitWebViewBaseDidRestoreScrollPosition(WebKitWebViewBase*);

void webkitWebViewBaseShowEmojiChooser(WebKitWebViewBase*, const WebCore::IntRect&, CompletionHandler<void(String)>&&);

void webkitWebViewBaseRequestPointerLock(WebKitWebViewBase*);
void webkitWebViewBaseDidLosePointerLock(WebKitWebViewBase*);

void webkitWebViewBaseSetInputMethodContext(WebKitWebViewBase*, WebKitInputMethodContext*);
WebKitInputMethodContext* webkitWebViewBaseGetInputMethodContext(WebKitWebViewBase*);
void webkitWebViewBaseSynthesizeCompositionKeyPress(WebKitWebViewBase*, const String& text, std::optional<Vector<WebCore::CompositionUnderline>>&&, std::optional<WebKit::EditingRange>&&);
void webkitWebViewBaseSynthesizeWheelEvent(WebKitWebViewBase*, const GdkEvent*, double deltaX, double deltaY, int x, int y, WheelEventPhase, WheelEventPhase momentumPhase, bool hasPreciseDeltas);

void webkitWebViewBaseMakeBlank(WebKitWebViewBase*, bool);
void webkitWebViewBasePageGrabbedTouch(WebKitWebViewBase*);
void webkitWebViewBaseSetShouldNotifyFocusEvents(WebKitWebViewBase*, bool);

void webkitWebViewBaseToplevelWindowIsActiveChanged(WebKitWebViewBase*, bool);
void webkitWebViewBaseToplevelWindowStateChanged(WebKitWebViewBase*, uint32_t, uint32_t);
void webkitWebViewBaseToplevelWindowMonitorChanged(WebKitWebViewBase*, GdkMonitor*);

void webkitWebViewBaseCallAfterNextPresentationUpdate(WebKitWebViewBase*, CompletionHandler<void()>&&);

#if USE(GTK4)
void webkitWebViewBaseSetPlugID(WebKitWebViewBase*, const String&);
#endif

WebKit::RendererBufferFormat webkitWebViewBaseGetRendererBufferFormat(WebKitWebViewBase*);
