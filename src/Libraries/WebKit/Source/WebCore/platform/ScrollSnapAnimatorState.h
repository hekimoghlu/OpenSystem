/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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

#include "FloatPoint.h"
#include "FloatSize.h"
#include "LayoutPoint.h"
#include "PlatformWheelEvent.h"
#include "ScrollAnimationMomentum.h"
#include "ScrollSnapOffsetsInfo.h"
#include "ScrollTypes.h"
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class Page;
class ScrollingEffectsController;
struct ScrollExtents;

enum class ScrollSnapState {
    Snapping,
    Gliding,
    DestinationReached,
    UserInteraction
};

class ScrollSnapAnimatorState {
    WTF_MAKE_TZONE_ALLOCATED(ScrollSnapAnimatorState);
public:
    ScrollSnapAnimatorState(ScrollingEffectsController& scrollController)
        : m_scrollController(scrollController)
    { }
    virtual ~ScrollSnapAnimatorState();

    const Vector<SnapOffset<LayoutUnit>>& snapOffsetsForAxis(ScrollEventAxis axis) const
    {
        return axis == ScrollEventAxis::Horizontal ? m_snapOffsetsInfo.horizontalSnapOffsets : m_snapOffsetsInfo.verticalSnapOffsets;
    }

    const LayoutScrollSnapOffsetsInfo& snapOffsetInfo() const { return m_snapOffsetsInfo; }
    void setSnapOffsetInfo(const LayoutScrollSnapOffsetsInfo& newInfo) { m_snapOffsetsInfo = newInfo; }

    ScrollSnapState currentState() const { return m_currentState; }

    std::optional<unsigned> activeSnapIndexForAxis(ScrollEventAxis axis) const
    {
        return axis == ScrollEventAxis::Horizontal ? m_activeSnapIndexX : m_activeSnapIndexY;
    }
    
    void setActiveSnapIndexForAxis(ScrollEventAxis, std::optional<unsigned>);

    std::optional<unsigned> closestSnapPointForOffset(ScrollEventAxis, ScrollOffset, const ScrollExtents&, float pageScale) const;
    float adjustedScrollDestination(ScrollEventAxis, FloatPoint destinationOffset, float velocity, std::optional<float> originalOffset, const ScrollExtents&, float pageScale) const;

    // returns true if an active snap index changed.
    bool resnapAfterLayout(ScrollOffset, const ScrollExtents&, float pageScale);

    bool setNearestScrollSnapIndexForOffset(ScrollOffset, const ScrollExtents&, float pageScale);

    // State transition helpers.
    // These return true if they start a new animation.
    bool transitionToSnapAnimationState(const ScrollExtents&, float pageScale, const FloatPoint& initialOffset);
    bool transitionToGlideAnimationState(const ScrollExtents&, float pageScale, const FloatPoint& initialOffset, const FloatSize& initialVelocity, const FloatSize& initialDelta);

    void transitionToUserInteractionState();
    void transitionToDestinationReachedState();

private:
    std::pair<float, std::optional<unsigned>> targetOffsetForStartOffset(ScrollEventAxis, const ScrollExtents&, float startOffset, FloatPoint predictedOffset, float pageScale, float initialDelta) const;
    bool setupAnimationForState(ScrollSnapState, const ScrollExtents&, float pageScale, const FloatPoint& initialOffset, const FloatSize& initialVelocity, const FloatSize& initialDelta);
    void teardownAnimationForState(ScrollSnapState);

    bool preserveCurrentTargetForAxis(ScrollEventAxis, ElementIdentifier);

    Vector<SnapOffset<LayoutUnit>> currentlySnappedOffsetsForAxis(ScrollEventAxis) const;
    UncheckedKeyHashSet<ElementIdentifier> currentlySnappedBoxes(const Vector<SnapOffset<LayoutUnit>>& horizontalOffsets, const Vector<SnapOffset<LayoutUnit>>& verticalOffsets) const;

    bool setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis, ScrollOffset, const ScrollExtents&, float pageScale);
    void updateCurrentlySnappedBoxes();

    void setActiveSnapIndexForAxisInternal(ScrollEventAxis axis, std::optional<unsigned> index)
    {
        if (axis == ScrollEventAxis::Horizontal)
            m_activeSnapIndexX = index;
        else
            m_activeSnapIndexY = index;
    }

    ScrollingEffectsController& m_scrollController;

    ScrollSnapState m_currentState { ScrollSnapState::UserInteraction };

    LayoutScrollSnapOffsetsInfo m_snapOffsetsInfo;
    
    std::optional<unsigned> m_activeSnapIndexX;
    std::optional<unsigned> m_activeSnapIndexY;
    UncheckedKeyHashSet<ElementIdentifier> m_currentlySnappedBoxes;
};

WTF::TextStream& operator<<(WTF::TextStream&, const ScrollSnapAnimatorState&);

} // namespace WebCore
