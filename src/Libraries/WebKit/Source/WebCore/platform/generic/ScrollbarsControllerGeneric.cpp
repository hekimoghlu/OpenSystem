/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#include "ScrollbarsControllerGeneric.h"

#include "ScrollAnimationKinetic.h"
#include "ScrollAnimationSmooth.h"
#include "ScrollableArea.h"
#include "ScrollbarTheme.h"

namespace WebCore {

static const Seconds overflowScrollbarsAnimationDuration { 1_s };
static const Seconds overflowScrollbarsAnimationHideDelay { 2_s };

std::unique_ptr<ScrollbarsController> ScrollbarsController::create(ScrollableArea& scrollableArea)
{
    return makeUnique<ScrollbarsControllerGeneric>(scrollableArea);
}

ScrollbarsControllerGeneric::ScrollbarsControllerGeneric(ScrollableArea& scrollableArea)
    : ScrollbarsController(scrollableArea)
    , m_overlayScrollbarAnimationTimer(*this, &ScrollbarsControllerGeneric::overlayScrollbarAnimationTimerFired)
{
}

ScrollbarsControllerGeneric::~ScrollbarsControllerGeneric() = default;

void ScrollbarsControllerGeneric::didAddVerticalScrollbar(Scrollbar* scrollbar)
{
    ScrollbarsController::didAddVerticalScrollbar(scrollbar);

    if (!scrollbar->isOverlayScrollbar())
        return;
    m_verticalOverlayScrollbar = scrollbar;
    if (!m_horizontalOverlayScrollbar)
        m_overlayScrollbarAnimationCurrent = 1;
    m_verticalOverlayScrollbar->setOpacity(m_overlayScrollbarAnimationCurrent);
    hideOverlayScrollbars();
}

void ScrollbarsControllerGeneric::didAddHorizontalScrollbar(Scrollbar* scrollbar)
{
    ScrollbarsController::didAddHorizontalScrollbar(scrollbar);

    if (!scrollbar->isOverlayScrollbar())
        return;
    m_horizontalOverlayScrollbar = scrollbar;
    if (!m_verticalOverlayScrollbar)
        m_overlayScrollbarAnimationCurrent = 1;
    m_horizontalOverlayScrollbar->setOpacity(m_overlayScrollbarAnimationCurrent);
    hideOverlayScrollbars();
}

void ScrollbarsControllerGeneric::willRemoveVerticalScrollbar(Scrollbar* scrollbar)
{
    if (m_verticalOverlayScrollbar != scrollbar)
        return;
    m_verticalOverlayScrollbar = nullptr;
    if (!m_horizontalOverlayScrollbar)
        m_overlayScrollbarAnimationCurrent = 0;
}

void ScrollbarsControllerGeneric::willRemoveHorizontalScrollbar(Scrollbar* scrollbar)
{
    if (m_horizontalOverlayScrollbar != scrollbar)
        return;
    m_horizontalOverlayScrollbar = nullptr;
    if (!m_verticalOverlayScrollbar)
        m_overlayScrollbarAnimationCurrent = 0;
}

void ScrollbarsControllerGeneric::updateOverlayScrollbarsOpacity()
{
    if (m_verticalOverlayScrollbar && m_overlayScrollbarAnimationCurrent != m_verticalOverlayScrollbar->opacity()) {
        m_verticalOverlayScrollbar->setOpacity(m_overlayScrollbarAnimationCurrent);
        if (m_verticalOverlayScrollbar->hoveredPart() == NoPart)
            m_verticalOverlayScrollbar->invalidate();
    }

    if (m_horizontalOverlayScrollbar && m_overlayScrollbarAnimationCurrent != m_horizontalOverlayScrollbar->opacity()) {
        m_horizontalOverlayScrollbar->setOpacity(m_overlayScrollbarAnimationCurrent);
        if (m_horizontalOverlayScrollbar->hoveredPart() == NoPart)
            m_horizontalOverlayScrollbar->invalidate();
    }
}

static inline double easeOutCubic(double t)
{
    double p = t - 1;
    return p * p * p + 1;
}

void ScrollbarsControllerGeneric::overlayScrollbarAnimationTimerFired()
{
    if (!m_horizontalOverlayScrollbar && !m_verticalOverlayScrollbar)
        return;
    if (m_overlayScrollbarsLocked)
        return;

    MonotonicTime currentTime = MonotonicTime::now();
    double progress = 1;
    if (currentTime < m_overlayScrollbarAnimationEndTime)
        progress = (currentTime - m_overlayScrollbarAnimationStartTime).value() / (m_overlayScrollbarAnimationEndTime - m_overlayScrollbarAnimationStartTime).value();
    progress = m_overlayScrollbarAnimationSource + (easeOutCubic(progress) * (m_overlayScrollbarAnimationTarget - m_overlayScrollbarAnimationSource));
    if (progress != m_overlayScrollbarAnimationCurrent) {
        m_overlayScrollbarAnimationCurrent = progress;
        updateOverlayScrollbarsOpacity();
    }

    if (m_overlayScrollbarAnimationCurrent != m_overlayScrollbarAnimationTarget) {
        static const double frameRate = 60;
        static const Seconds tickTime = 1_s / frameRate;
        static const Seconds minimumTimerInterval = 1_ms;
        Seconds deltaToNextFrame = std::max(tickTime - (MonotonicTime::now() - currentTime), minimumTimerInterval);
        m_overlayScrollbarAnimationTimer.startOneShot(deltaToNextFrame);
    } else
        hideOverlayScrollbars();
}

void ScrollbarsControllerGeneric::showOverlayScrollbars()
{
    if (m_overlayScrollbarsLocked)
        return;

    if (m_overlayScrollbarAnimationTimer.isActive() && m_overlayScrollbarAnimationTarget == 1)
        return;
    m_overlayScrollbarAnimationTimer.stop();

    if (!m_horizontalOverlayScrollbar && !m_verticalOverlayScrollbar)
        return;

    m_overlayScrollbarAnimationSource = m_overlayScrollbarAnimationCurrent;
    m_overlayScrollbarAnimationTarget = 1;
    if (m_overlayScrollbarAnimationTarget != m_overlayScrollbarAnimationCurrent) {
        m_overlayScrollbarAnimationStartTime = MonotonicTime::now();
        m_overlayScrollbarAnimationEndTime = m_overlayScrollbarAnimationStartTime + overflowScrollbarsAnimationDuration;
        m_overlayScrollbarAnimationTimer.startOneShot(0_s);
    } else
        hideOverlayScrollbars();
}

void ScrollbarsControllerGeneric::hideOverlayScrollbars()
{
    if (m_overlayScrollbarAnimationTimer.isActive() && !m_overlayScrollbarAnimationTarget)
        return;
    m_overlayScrollbarAnimationTimer.stop();

    if (!m_horizontalOverlayScrollbar && !m_verticalOverlayScrollbar)
        return;

    m_overlayScrollbarAnimationSource = m_overlayScrollbarAnimationCurrent;
    m_overlayScrollbarAnimationTarget = 0;
    if (m_overlayScrollbarAnimationTarget == m_overlayScrollbarAnimationCurrent)
        return;
    m_overlayScrollbarAnimationStartTime = MonotonicTime::now() + overflowScrollbarsAnimationHideDelay;
    m_overlayScrollbarAnimationEndTime = m_overlayScrollbarAnimationStartTime + overflowScrollbarsAnimationDuration + overflowScrollbarsAnimationHideDelay;
    m_overlayScrollbarAnimationTimer.startOneShot(overflowScrollbarsAnimationHideDelay);
}

void ScrollbarsControllerGeneric::mouseEnteredContentArea()
{
    showOverlayScrollbars();
}

void ScrollbarsControllerGeneric::mouseExitedContentArea()
{
    hideOverlayScrollbars();
}

void ScrollbarsControllerGeneric::mouseMovedInContentArea()
{
    showOverlayScrollbars();
}

void ScrollbarsControllerGeneric::contentAreaDidShow()
{
    showOverlayScrollbars();
}

void ScrollbarsControllerGeneric::contentAreaDidHide()
{
    if (m_overlayScrollbarsLocked)
        return;
    m_overlayScrollbarAnimationTimer.stop();
    if (m_overlayScrollbarAnimationCurrent) {
        m_overlayScrollbarAnimationCurrent = 0;
        updateOverlayScrollbarsOpacity();
    }
}

void ScrollbarsControllerGeneric::notifyContentAreaScrolled(const FloatSize&)
{
    showOverlayScrollbars();
}

void ScrollbarsControllerGeneric::lockOverlayScrollbarStateToHidden(bool shouldLockState)
{
    if (m_overlayScrollbarsLocked == shouldLockState)
        return;
    m_overlayScrollbarsLocked = shouldLockState;

    if (!m_horizontalOverlayScrollbar && !m_verticalOverlayScrollbar)
        return;

    if (m_overlayScrollbarsLocked) {
        m_overlayScrollbarAnimationTimer.stop();
        if (m_horizontalOverlayScrollbar)
            m_horizontalOverlayScrollbar->setOpacity(0);
        if (m_verticalOverlayScrollbar)
            m_verticalOverlayScrollbar->setOpacity(0);
    } else {
        if (m_overlayScrollbarAnimationCurrent == 1)
            updateOverlayScrollbarsOpacity();
        else
            showOverlayScrollbars();
    }
}

} // namespace WebCore

