/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "RenderProgress.h"

#include "HTMLProgressElement.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderStyleInlines.h"
#include "RenderTheme.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderProgress);

RenderProgress::RenderProgress(HTMLElement& element, RenderStyle&& style)
    : RenderBlockFlow(Type::Progress, element, WTFMove(style))
    , m_position(HTMLProgressElement::InvalidPosition)
    , m_animationTimer(*this, &RenderProgress::animationTimerFired)
{
    ASSERT(isRenderProgress());
}

RenderProgress::~RenderProgress() = default;

void RenderProgress::updateFromElement()
{
    HTMLProgressElement* element = progressElement();
    if (m_position == element->position())
        return;
    m_position = element->position();

    updateAnimationState();
    repaint();
    RenderBlockFlow::updateFromElement();
}

RenderBox::LogicalExtentComputedValues RenderProgress::computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const
{
    auto computedValues = RenderBox::computeLogicalHeight(logicalHeight, logicalTop);
    LayoutRect frame = frameRect();
    if (isHorizontalWritingMode())
        frame.setHeight(computedValues.m_extent);
    else
        frame.setWidth(computedValues.m_extent);
    IntSize frameSize = theme().progressBarRectForBounds(*this, snappedIntRect(frame)).size();
    computedValues.m_extent = isHorizontalWritingMode() ? frameSize.height() : frameSize.width();
    return computedValues;
}

double RenderProgress::animationProgress() const
{
    auto duration = theme().animationDurationForProgressBar();
    ASSERT(duration > 0_s);
    return m_animating ? (fmod((MonotonicTime::now() - m_animationStartTime).seconds(), duration.seconds()) / duration.seconds()) : 0;
}

bool RenderProgress::isDeterminate() const
{
    return (HTMLProgressElement::IndeterminatePosition != position()
            && HTMLProgressElement::InvalidPosition != position());
}

void RenderProgress::animationTimerFired()
{
    // FIXME: Ideally obtaining the repeat interval from Page is not RenderTheme-specific, but it
    // currently is as it also determines whether we animate at all.
    auto repeatInterval = theme().animationRepeatIntervalForProgressBar(*this);

    repaint();
    if (!m_animationTimer.isActive() && m_animating)
        m_animationTimer.startOneShot(repeatInterval);
}

void RenderProgress::updateAnimationState()
{
    auto repeatInterval = theme().animationRepeatIntervalForProgressBar(*this);

    bool animating = style().hasUsedAppearance() && repeatInterval > 0_s && !isDeterminate();
    if (animating == m_animating)
        return;

    m_animating = animating;
    if (m_animating) {
        m_animationStartTime = MonotonicTime::now();
        m_animationTimer.startOneShot(repeatInterval);
    } else
        m_animationTimer.stop();
}

HTMLProgressElement* RenderProgress::progressElement() const
{
    if (!element())
        return nullptr;

    if (auto* progressElement = dynamicDowncast<HTMLProgressElement>(*element()))
        return progressElement;

    ASSERT(element()->shadowHost());
    return downcast<HTMLProgressElement>(element()->shadowHost());
}    

} // namespace WebCore

