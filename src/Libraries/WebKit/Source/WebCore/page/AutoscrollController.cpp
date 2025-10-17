/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#include "AutoscrollController.h"

#include "EventHandler.h"
#include "HitTestResult.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "RenderBox.h"
#include "RenderListBox.h"
#include "RenderView.h"
#include "ScrollView.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AutoscrollController);

// Delay time in second for start autoscroll if pointer is in border edge of scrollable element.
static const Seconds autoscrollDelay { 200_ms };

#if ENABLE(PAN_SCROLLING)
static LocalFrame* getMainFrame(LocalFrame* frame)
{
    Page* page = frame->page();
    return page && dynamicDowncast<LocalFrame>(page->mainFrame()) ? dynamicDowncast<LocalFrame>(page->mainFrame()) : 0;
}
#endif

AutoscrollController::AutoscrollController()
    : m_autoscrollTimer(*this, &AutoscrollController::autoscrollTimerFired)
{
}

RenderBox* AutoscrollController::autoscrollRenderer() const
{
    return m_autoscrollRenderer.get();
}

bool AutoscrollController::autoscrollInProgress() const
{
    return m_autoscrollType == AutoscrollType::Selection;
}

void AutoscrollController::startAutoscrollForSelection(RenderObject* renderer)
{
    // We don't want to trigger the autoscroll or the panScroll if it's already active
    if (m_autoscrollTimer.isActive())
        return;
    auto* scrollable = RenderBox::findAutoscrollable(renderer);
    if (!scrollable)
        scrollable = renderer->isRenderListBox() ? downcast<RenderListBox>(renderer) : nullptr;
    if (!scrollable)
        return;
    m_autoscrollType = AutoscrollType::Selection;
    m_autoscrollRenderer = WeakPtr { *scrollable };
    startAutoscrollTimer();
}

void AutoscrollController::stopAutoscrollTimer(bool rendererIsBeingDestroyed)
{
    CheckedPtr scrollable = m_autoscrollRenderer.get();

    m_autoscrollTimer.stop();
    m_autoscrollRenderer = nullptr;

    if (!scrollable)
        return;

    RefPtr frame = scrollable->document().frame();
    if (autoscrollInProgress() && frame && frame->eventHandler().mouseDownWasInSubframe()) {
        if (RefPtr subframe = dynamicDowncast<LocalFrame>(frame->checkedEventHandler()->subframeForTargetNode(frame->eventHandler().mousePressNode())))
            subframe->checkedEventHandler()->stopAutoscrollTimer(rendererIsBeingDestroyed);
        return;
    }

    if (!rendererIsBeingDestroyed)
        scrollable->stopAutoscroll();

#if ENABLE(PAN_SCROLLING)
    if (panScrollInProgress()) {
        Ref frameView = scrollable->view().frameView();
        frameView->removePanScrollIcon();
        frameView->setCursor(pointerCursor());
    }
#endif

    m_autoscrollType = AutoscrollType::None;

#if ENABLE(PAN_SCROLLING)
    // If we're not in the top frame we notify it that we are not doing a panScroll any more.
    if (RefPtr localFrame = (frame && !frame->isMainFrame()) ? dynamicDowncast<LocalFrame>(frame->mainFrame()) : nullptr)
        localFrame->checkedEventHandler()->didPanScrollStop();
#endif
}

void AutoscrollController::updateAutoscrollRenderer()
{
    if (!m_autoscrollRenderer)
        return;

    RenderObject* renderer = m_autoscrollRenderer.get();

#if ENABLE(PAN_SCROLLING)
    constexpr OptionSet<HitTestRequest::Type> hitType { HitTestRequest::Type::ReadOnly, HitTestRequest::Type::Active, HitTestRequest::Type::AllowChildFrameContent };
    HitTestResult hitTest = m_autoscrollRenderer->protectedFrame()->checkedEventHandler()->hitTestResultAtPoint(m_panScrollStartPos, hitType);

    if (auto* nodeAtPoint = hitTest.innerNode())
        renderer = nodeAtPoint->renderer();
#endif

    while (renderer && !(is<RenderBox>(*renderer) && downcast<RenderBox>(*renderer).canAutoscroll()))
        renderer = renderer->parent();

    if (!is<RenderBox>(renderer)) {
        m_autoscrollRenderer = nullptr;
        return;
    }

    m_autoscrollRenderer = WeakPtr { downcast<RenderBox>(*renderer) };
}

void AutoscrollController::updateDragAndDrop(Node* dropTargetNode, const IntPoint& eventPosition, WallTime eventTime)
{
    IntSize offset;
    auto findDragAndDropScroller = [&]() -> RenderBox* {
        if (!dropTargetNode)
            return nullptr;

        CheckedPtr scrollable = RenderBox::findAutoscrollable(dropTargetNode->renderer());
        if (!scrollable)
            return nullptr;

        RefPtr page = scrollable->frame().page();
        if (!page || !page->settings().autoscrollForDragAndDropEnabled())
            return nullptr;

        offset = scrollable->calculateAutoscrollDirection(eventPosition);
        if (offset.isZero())
            return nullptr;

        return scrollable.get();
    };
    
    CheckedPtr scrollable = findDragAndDropScroller();
    if (!scrollable) {
        stopAutoscrollTimer();
        return;
    }

    m_dragAndDropAutoscrollReferencePosition = eventPosition + offset;

    if (m_autoscrollType == AutoscrollType::None) {
        m_autoscrollType = AutoscrollType::DragAndDrop;
        m_autoscrollRenderer = WeakPtr { *scrollable };
        m_dragAndDropAutoscrollStartTime = eventTime;
        startAutoscrollTimer();
    } else if (m_autoscrollRenderer != scrollable.get()) {
        m_dragAndDropAutoscrollStartTime = eventTime;
        m_autoscrollRenderer = WeakPtr { *scrollable };
    }
}

#if ENABLE(PAN_SCROLLING)
void AutoscrollController::didPanScrollStart()
{
    m_autoscrollType = AutoscrollType::Pan;
}

void AutoscrollController::didPanScrollStop()
{
    m_autoscrollType = AutoscrollType::None;
}

void AutoscrollController::handleMouseReleaseEvent(const PlatformMouseEvent& mouseEvent)
{
    switch (m_autoscrollType) {
    case AutoscrollType::Pan:
        if (mouseEvent.button() == MouseButton::Middle)
            m_autoscrollType = AutoscrollType::PanCanStop;
        break;
    case AutoscrollType::PanCanStop:
        stopAutoscrollTimer();
        break;
    default:
        break;
    }
}

bool AutoscrollController::panScrollInProgress() const
{
    return m_autoscrollType == AutoscrollType::Pan || m_autoscrollType == AutoscrollType::PanCanStop;
}

void AutoscrollController::startPanScrolling(RenderBox& scrollable, const IntPoint& lastKnownMousePosition)
{
    // We don't want to trigger the autoscroll or the panScroll if it's already active
    if (m_autoscrollTimer.isActive())
        return;

    m_autoscrollType = AutoscrollType::Pan;
    m_autoscrollRenderer = WeakPtr { scrollable };
    m_panScrollStartPos = lastKnownMousePosition;

    if (RefPtr view = scrollable.frame().view())
        view->addPanScrollIcon(lastKnownMousePosition);

    scrollable.protectedFrame()->checkedEventHandler()->didPanScrollStart();
    startAutoscrollTimer();
}
#else
bool AutoscrollController::panScrollInProgress() const
{
    return false;
}
#endif

void AutoscrollController::autoscrollTimerFired()
{
    if (!m_autoscrollRenderer) {
        stopAutoscrollTimer();
        return;
    }

    Ref frame = m_autoscrollRenderer->frame();
    switch (m_autoscrollType) {
    case AutoscrollType::DragAndDrop:
        if (WallTime::now() - m_dragAndDropAutoscrollStartTime > autoscrollDelay)
            CheckedRef { *m_autoscrollRenderer }->autoscroll(m_dragAndDropAutoscrollReferencePosition);
        break;
    case AutoscrollType::Selection: {
        if (!frame->checkedEventHandler()->shouldUpdateAutoscroll()) {
            stopAutoscrollTimer();
            return;
        }
#if ENABLE(DRAG_SUPPORT)
        frame->checkedEventHandler()->updateSelectionForMouseDrag();
#endif
        CheckedRef { *m_autoscrollRenderer }->autoscroll(frame->checkedEventHandler()->targetPositionInWindowForSelectionAutoscroll());
        break;
    }
    case AutoscrollType::None:
        break;
#if ENABLE(PAN_SCROLLING)
    case AutoscrollType::PanCanStop:
    case AutoscrollType::Pan:
        // we verify that the main frame hasn't received the order to stop the panScroll
        if (RefPtr mainFrame = getMainFrame(frame.ptr())) {
            if (!mainFrame->checkedEventHandler()->panScrollInProgress()) {
                stopAutoscrollTimer();
                return;
            }
        }
        if (RefPtr view = frame->view())
            updatePanScrollState(view.get(), frame->checkedEventHandler()->lastKnownMousePosition());
        CheckedRef { *m_autoscrollRenderer }->panScroll(m_panScrollStartPos);
        break;
#endif
    }
}

void AutoscrollController::startAutoscrollTimer()
{
    m_autoscrollTimer.startRepeating(autoscrollInterval);
}

#if ENABLE(PAN_SCROLLING)
void AutoscrollController::updatePanScrollState(LocalFrameView* view, const IntPoint& lastKnownMousePosition)
{
    // At the original click location we draw a 4 arrowed icon. Over this icon there won't be any scroll
    // So we don't want to change the cursor over this area
    bool east = m_panScrollStartPos.x() < (lastKnownMousePosition.x() - ScrollView::noPanScrollRadius);
    bool west = m_panScrollStartPos.x() > (lastKnownMousePosition.x() + ScrollView::noPanScrollRadius);
    bool north = m_panScrollStartPos.y() > (lastKnownMousePosition.y() + ScrollView::noPanScrollRadius);
    bool south = m_panScrollStartPos.y() < (lastKnownMousePosition.y() - ScrollView::noPanScrollRadius);

    if (m_autoscrollType == AutoscrollType::Pan && (east || west || north || south))
        m_autoscrollType = AutoscrollType::PanCanStop;

    if (north) {
        if (east)
            view->setCursor(northEastPanningCursor());
        else if (west)
            view->setCursor(northWestPanningCursor());
        else
            view->setCursor(northPanningCursor());
    } else if (south) {
        if (east)
            view->setCursor(southEastPanningCursor());
        else if (west)
            view->setCursor(southWestPanningCursor());
        else
            view->setCursor(southPanningCursor());
    } else if (east)
        view->setCursor(eastPanningCursor());
    else if (west)
        view->setCursor(westPanningCursor());
    else
        view->setCursor(middlePanningCursor());
}
#endif

} // namespace WebCore
