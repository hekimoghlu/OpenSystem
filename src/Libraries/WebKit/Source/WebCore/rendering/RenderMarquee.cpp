/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include "RenderMarquee.h"

#include "HTMLMarqueeElement.h"
#include "HTMLNames.h"
#include "LocalFrameView.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include "RenderLayerScrollableArea.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderMarquee);

using namespace HTMLNames;

RenderMarquee::RenderMarquee(RenderLayer* layer)
    : m_layer(layer)
    , m_timer(*this, &RenderMarquee::timerFired)
{
    ASSERT(layer);
    ASSERT(layer->scrollableArea());
    layer->scrollableArea()->setScrollClamping(ScrollClamping::Unclamped);
}

RenderMarquee::~RenderMarquee() = default;

int RenderMarquee::marqueeSpeed() const
{
    int result = m_layer->renderer().style().marqueeSpeed();
    if (auto* marquee = dynamicDowncast<HTMLMarqueeElement>(m_layer->renderer().element()))
        result = std::max(result, marquee->minimumDelay());
    return result;
}

static MarqueeDirection reverseDirection(MarqueeDirection direction)
{
    switch (direction) {
    case MarqueeDirection::Auto:
        return MarqueeDirection::Auto;
    case MarqueeDirection::Left:
        return MarqueeDirection::Right;
    case MarqueeDirection::Right:
        return MarqueeDirection::Left;
    case MarqueeDirection::Up:
        return MarqueeDirection::Down;
    case MarqueeDirection::Down:
        return MarqueeDirection::Up;
    case MarqueeDirection::Backward:
        return MarqueeDirection::Forward;
    case MarqueeDirection::Forward:
        return MarqueeDirection::Backward;
    }
    return MarqueeDirection::Auto;
}

MarqueeDirection RenderMarquee::direction() const
{
    // FIXME: Support the CSS3 "auto" value for determining the direction of the marquee.
    // For now just map MarqueeDirection::Auto to MarqueeDirection::Backward
    MarqueeDirection result = m_layer->renderer().style().marqueeDirection();
    WritingMode writingMode = m_layer->renderer().writingMode();
    if (result == MarqueeDirection::Auto)
        result = MarqueeDirection::Backward;
    if (result == MarqueeDirection::Forward)
        result = (writingMode.isBidiLTR()) ? MarqueeDirection::Right : MarqueeDirection::Left;
    if (result == MarqueeDirection::Backward)
        result = (writingMode.isBidiLTR()) ? MarqueeDirection::Left : MarqueeDirection::Right;

    // Now we have the real direction.  Next we check to see if the increment is negative.
    // If so, then we reverse the direction.
    Length increment = m_layer->renderer().style().marqueeIncrement();
    if (increment.isNegative())
        result = reverseDirection(result);
    
    return result;
}

bool RenderMarquee::isHorizontal() const
{
    return direction() == MarqueeDirection::Left || direction() == MarqueeDirection::Right;
}

int RenderMarquee::computePosition(MarqueeDirection dir, bool stopAtContentEdge)
{
    RenderBox* box = m_layer->renderBox();
    ASSERT(box);
    auto& boxStyle = box->style();
    if (isHorizontal()) {
        bool ltr = boxStyle.isLeftToRightDirection();
        LayoutUnit clientWidth = box->clientWidth();
        LayoutUnit contentWidth = ltr ? box->maxPreferredLogicalWidth() : box->minPreferredLogicalWidth();
        if (ltr)
            contentWidth += (box->paddingRight() - box->borderLeft());
        else {
            contentWidth = box->width() - contentWidth;
            contentWidth += (box->paddingLeft() - box->borderRight());
        }
        if (dir == MarqueeDirection::Right) {
            if (stopAtContentEdge)
                return std::max<LayoutUnit>(0, ltr ? (contentWidth - clientWidth) : (clientWidth - contentWidth));

            return ltr ? contentWidth : clientWidth;
        }

        if (stopAtContentEdge)
            return std::min<LayoutUnit>(0, ltr ? (contentWidth - clientWidth) : (clientWidth - contentWidth));

        return ltr ? -clientWidth : -contentWidth;
    }

    // Vertical
    int contentHeight = box->layoutOverflowRect().maxY() - box->borderTop() + box->paddingBottom();
    int clientHeight = roundToInt(box->clientHeight());
    if (dir == MarqueeDirection::Up) {
        if (stopAtContentEdge)
            return std::min(contentHeight - clientHeight, 0);

        return -clientHeight;
    }

    if (stopAtContentEdge)
        return std::max(contentHeight - clientHeight, 0);

    return contentHeight;
}

void RenderMarquee::start()
{
    if (m_timer.isActive() || m_layer->renderer().style().marqueeIncrement().isZero())
        return;

    auto* scrollableArea = m_layer->scrollableArea();
    ASSERT(scrollableArea);

    auto details = ScrollPositionChangeOptions::createProgrammaticUnclamped();
    if (!m_suspended && !m_stopped) {
        if (isHorizontal())
            scrollableArea->scrollToOffset(ScrollOffset(m_start, 0), details);
        else
            scrollableArea->scrollToOffset(ScrollOffset(0, m_start), details);
    } else {
        m_suspended = false;
        m_stopped = false;
    }

    m_timer.startRepeating(1_ms * speed());
}

void RenderMarquee::suspend()
{
    m_timer.stop();
    m_suspended = true;
}

void RenderMarquee::stop()
{
    m_timer.stop();
    m_stopped = true;
}

void RenderMarquee::updateMarqueePosition()
{
    bool activate = (m_totalLoops <= 0 || m_currentLoop < m_totalLoops);
    if (activate) {
        MarqueeBehavior behavior = m_layer->renderer().style().marqueeBehavior();
        m_start = computePosition(direction(), behavior == MarqueeBehavior::Alternate);
        m_end = computePosition(reverseDirection(direction()), behavior == MarqueeBehavior::Alternate || behavior == MarqueeBehavior::Slide);
        if (!m_stopped)
            start();
    }
}

void RenderMarquee::updateMarqueeStyle()
{
    auto& style = m_layer->renderer().style();
    
    if (m_direction != style.marqueeDirection() || (m_totalLoops != style.marqueeLoopCount() && m_currentLoop >= m_totalLoops))
        m_currentLoop = 0; // When direction changes or our loopCount is a smaller number than our current loop, reset our loop.
    
    m_totalLoops = style.marqueeLoopCount();
    m_direction = style.marqueeDirection();
    
    if (m_layer->renderer().isHTMLMarquee()) {
        // Hack for WinIE.  In WinIE, a value of 0 or lower for the loop count for SLIDE means to only do
        // one loop.
        if (m_totalLoops <= 0 && style.marqueeBehavior() == MarqueeBehavior::Slide)
            m_totalLoops = 1;
    }
    
    if (speed() != marqueeSpeed()) {
        m_speed = marqueeSpeed();
        if (m_timer.isActive())
            m_timer.startRepeating(1_ms * speed());
    }
    
    // Check the loop count to see if we should now stop.
    bool activate = (m_totalLoops <= 0 || m_currentLoop < m_totalLoops);
    if (activate && !m_timer.isActive())
        m_layer->renderer().setNeedsLayout();
    else if (!activate && m_timer.isActive())
        m_timer.stop();
}

void RenderMarquee::timerFired()
{
    if (m_layer->renderer().view().needsLayout())
        return;

    auto* scrollableArea = m_layer->scrollableArea();
    ASSERT(scrollableArea);

    if (m_reset) {
        m_reset = false;
        if (isHorizontal())
            scrollableArea->scrollToXOffset(m_start);
        else
            scrollableArea->scrollToYOffset(m_start);
        return;
    }
    
    const RenderStyle& style = m_layer->renderer().style();
    
    int endPoint = m_end;
    int range = m_end - m_start;
    int newPos;
    if (range == 0)
        newPos = m_end;
    else {  
        bool addIncrement = direction() == MarqueeDirection::Up || direction() == MarqueeDirection::Left;
        bool isReversed = style.marqueeBehavior() == MarqueeBehavior::Alternate && m_currentLoop % 2;
        if (isReversed) {
            // We're going in the reverse direction.
            endPoint = m_start;
            range = -range;
            addIncrement = !addIncrement;
        }
        bool positive = range > 0;
        int clientSize = (isHorizontal() ? roundToInt(m_layer->renderBox()->clientWidth()) : roundToInt(m_layer->renderBox()->clientHeight()));
        int increment = std::abs(intValueForLength(m_layer->renderer().style().marqueeIncrement(), clientSize));
        int currentPos = (isHorizontal() ? scrollableArea->scrollOffset().x() : scrollableArea->scrollOffset().y());
        newPos =  currentPos + (addIncrement ? increment : -increment);
        if (positive)
            newPos = std::min(newPos, endPoint);
        else
            newPos = std::max(newPos, endPoint);
    }

    if (newPos == endPoint) {
        m_currentLoop++;
        if (m_totalLoops > 0 && m_currentLoop >= m_totalLoops)
            m_timer.stop();
        else if (style.marqueeBehavior() != MarqueeBehavior::Alternate)
            m_reset = true;
    }
    
    if (isHorizontal())
        scrollableArea->scrollToXOffset(newPos);
    else
        scrollableArea->scrollToYOffset(newPos);
}

} // namespace WebCore
