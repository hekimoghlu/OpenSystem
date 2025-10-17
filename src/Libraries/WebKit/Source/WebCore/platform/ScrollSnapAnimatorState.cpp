/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#include "ScrollSnapAnimatorState.h"

#include "Logging.h"
#include "ScrollExtents.h"
#include "ScrollingEffectsController.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollSnapAnimatorState);

ScrollSnapAnimatorState::~ScrollSnapAnimatorState() = default;

bool ScrollSnapAnimatorState::transitionToSnapAnimationState(const ScrollExtents& scrollExtents, float pageScale, const FloatPoint& initialOffset)
{
    return setupAnimationForState(ScrollSnapState::Snapping, scrollExtents, pageScale, initialOffset, { }, { });
}

bool ScrollSnapAnimatorState::transitionToGlideAnimationState(const ScrollExtents& scrollExtents, float pageScale, const FloatPoint& initialOffset, const FloatSize& initialVelocity, const FloatSize& initialDelta)
{
    return setupAnimationForState(ScrollSnapState::Gliding, scrollExtents, pageScale, initialOffset, initialVelocity, initialDelta);
}

bool ScrollSnapAnimatorState::setupAnimationForState(ScrollSnapState state, const ScrollExtents& scrollExtents, float pageScale, const FloatPoint& initialOffset, const FloatSize& initialVelocity, const FloatSize& initialDelta)
{
    ASSERT(state == ScrollSnapState::Snapping || state == ScrollSnapState::Gliding);
    if (m_currentState == state)
        return false;

    bool animating = m_scrollController.startMomentumScrollWithInitialVelocity(initialOffset, initialVelocity, initialDelta, [&](const FloatPoint& targetOffset) {
        float targetOffsetX, targetOffsetY;
        std::tie(targetOffsetX, m_activeSnapIndexX) = targetOffsetForStartOffset(ScrollEventAxis::Horizontal, scrollExtents, initialOffset.x(), targetOffset, pageScale, initialDelta.width());
        std::tie(targetOffsetY, m_activeSnapIndexY) = targetOffsetForStartOffset(ScrollEventAxis::Vertical, scrollExtents, initialOffset.y(), targetOffset, pageScale, initialDelta.height());
        LOG_WITH_STREAM(ScrollAnimations, stream << "ScrollSnapAnimatorState::setupAnimationForState() - target offset " << targetOffset << " modified to " << FloatPoint(targetOffsetX, targetOffsetY));
        return FloatPoint { targetOffsetX, targetOffsetY };
    });

    if (!animating)
        return false;

    m_currentState = state;
    return animating;
}

std::optional<unsigned> ScrollSnapAnimatorState::closestSnapPointForOffset(ScrollEventAxis axis, ScrollOffset scrollOffset, const ScrollExtents& scrollExtents, float pageScale) const
{
    LayoutPoint layoutScrollOffset(scrollOffset.x() / pageScale, scrollOffset.y() / pageScale);

    auto snapOffsets = snapOffsetsForAxis(axis);
    LayoutSize viewportSize(scrollExtents.viewportSize);
    std::optional<unsigned> activeIndex;
    if (snapOffsets.size())
        activeIndex = snapOffsetInfo().closestSnapOffset(axis, viewportSize, layoutScrollOffset, 0).second;

    return activeIndex;
}

float ScrollSnapAnimatorState::adjustedScrollDestination(ScrollEventAxis axis, FloatPoint destinationOffset, float velocity, std::optional<float> originalOffset, const ScrollExtents& scrollExtents, float pageScale) const
{
    auto snapOffsets = snapOffsetsForAxis(axis);
    if (!snapOffsets.size())
        return valueForAxis(destinationOffset, axis);

    std::optional<LayoutUnit> originalOffsetInLayoutUnits;
    if (originalOffset)
        originalOffsetInLayoutUnits = LayoutUnit(*originalOffset / pageScale);
    LayoutSize viewportSize(scrollExtents.viewportSize);
    LayoutPoint layoutDestinationOffset(destinationOffset.x() / pageScale, destinationOffset.y() / pageScale);
    LayoutUnit offset = snapOffsetInfo().closestSnapOffset(axis, viewportSize, layoutDestinationOffset, velocity, originalOffsetInLayoutUnits).first;
    return offset * pageScale;
}

// Returns whether the snap point is changed or not
bool ScrollSnapAnimatorState::preserveCurrentTargetForAxis(ScrollEventAxis axis, ElementIdentifier boxID)
{
    auto snapOffsets = snapOffsetsForAxis(axis);
    
    auto found = std::find_if(snapOffsets.begin(), snapOffsets.end(), [boxID](SnapOffset<LayoutUnit> p) -> bool {
        return *p.snapTargetID == boxID;
    });
    if (found == snapOffsets.end()) {
        setActiveSnapIndexForAxis(axis, std::nullopt);
        return false;
    }
    
    setActiveSnapIndexForAxis(axis, std::distance(snapOffsets.begin(), found));
    return true;
}

Vector<SnapOffset<LayoutUnit>> ScrollSnapAnimatorState::currentlySnappedOffsetsForAxis(ScrollEventAxis axis) const
{
    Vector<SnapOffset<LayoutUnit>> currentlySnappedOffsets;
    auto snapOffsets = snapOffsetsForAxis(axis);
    auto activeIndex = activeSnapIndexForAxis(axis);
    
    if (activeIndex && *activeIndex < snapOffsets.size())
        currentlySnappedOffsets.append(snapOffsets[*activeIndex]);
    return currentlySnappedOffsets;
}

UncheckedKeyHashSet<ElementIdentifier> ScrollSnapAnimatorState::currentlySnappedBoxes(const Vector<SnapOffset<LayoutUnit>>& horizontalOffsets, const Vector<SnapOffset<LayoutUnit>>& verticalOffsets) const
{
    UncheckedKeyHashSet<ElementIdentifier> snappedBoxIDs;
        
    for (auto offset : horizontalOffsets) {
        if (!offset.snapTargetID)
            continue;
        snappedBoxIDs.add(*offset.snapTargetID);
        for (auto i : offset.snapAreaIndices)
            snappedBoxIDs.add(m_snapOffsetsInfo.snapAreasIDs[i]);
    }
    
    for (auto offset : verticalOffsets) {
        if (!offset.snapTargetID)
            continue;
        snappedBoxIDs.add(*offset.snapTargetID);
        for (auto i : offset.snapAreaIndices)
            snappedBoxIDs.add(m_snapOffsetsInfo.snapAreasIDs[i]);
    }
    return snappedBoxIDs;
}

void ScrollSnapAnimatorState::setActiveSnapIndexForAxis(ScrollEventAxis axis, std::optional<unsigned> index)
{
    setActiveSnapIndexForAxisInternal(axis, index);
    updateCurrentlySnappedBoxes();
}

void ScrollSnapAnimatorState::updateCurrentlySnappedBoxes()
{
    auto horizontalOffsets = currentlySnappedOffsetsForAxis(ScrollEventAxis::Horizontal);
    auto verticalOffsets = currentlySnappedOffsetsForAxis(ScrollEventAxis::Vertical);

    m_currentlySnappedBoxes = currentlySnappedBoxes(horizontalOffsets, verticalOffsets);
}

static ElementIdentifier chooseBoxToResnapTo(const UncheckedKeyHashSet<ElementIdentifier>& snappedBoxes, const Vector<SnapOffset<LayoutUnit>>& horizontalOffsets, const Vector<SnapOffset<LayoutUnit>>& verticalOffsets)
{
    ASSERT(snappedBoxes.size());

    auto found = std::find_if(horizontalOffsets.begin(), horizontalOffsets.end(), [&snappedBoxes](SnapOffset<LayoutUnit> p) -> bool {
        return snappedBoxes.contains(*p.snapTargetID) && p.isFocused;
    });
    if (found != horizontalOffsets.end())
        return *found->snapTargetID;
    
    found = std::find_if(verticalOffsets.begin(), verticalOffsets.end(), [&snappedBoxes](SnapOffset<LayoutUnit> p) -> bool {
        return snappedBoxes.contains(*p.snapTargetID) && p.isFocused;
    });
    if (found != verticalOffsets.end())
        return *found->snapTargetID;
    
    return *snappedBoxes.begin();
}

bool ScrollSnapAnimatorState::resnapAfterLayout(ScrollOffset scrollOffset, const ScrollExtents& scrollExtents, float pageScale)
{
    bool snapPointChanged = false;
    auto activeHorizontalIndex = activeSnapIndexForAxis(ScrollEventAxis::Horizontal);
    auto activeVerticalIndex = activeSnapIndexForAxis(ScrollEventAxis::Vertical);
    auto snapOffsetsVertical = snapOffsetsForAxis(ScrollEventAxis::Vertical);
    auto snapOffsetsHorizontal = snapOffsetsForAxis(ScrollEventAxis::Horizontal);
    
    auto previouslySnappedBoxes = std::exchange(m_currentlySnappedBoxes, { });
    
    // Check if we need to set the current indices
    if (!activeVerticalIndex || *activeVerticalIndex >= snapOffsetsForAxis(ScrollEventAxis::Vertical).size()) {
        snapPointChanged |= setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis::Vertical, scrollOffset, scrollExtents, pageScale);
        activeVerticalIndex = activeSnapIndexForAxis(ScrollEventAxis::Vertical);
    }
    if (!activeHorizontalIndex || *activeHorizontalIndex >= snapOffsetsForAxis(ScrollEventAxis::Horizontal).size()) {
        snapPointChanged |= setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis::Horizontal, scrollOffset, scrollExtents, pageScale);
        activeHorizontalIndex = activeSnapIndexForAxis(ScrollEventAxis::Horizontal);
    }

    updateCurrentlySnappedBoxes();
    LOG_WITH_STREAM(ScrollSnap, stream << "ScrollSnapAnimatorState::resnapAfterLayout() - previouslySnappedBoxes " << previouslySnappedBoxes << " m_currentlySnappedBoxes " << m_currentlySnappedBoxes);

    auto wasSnappedToMultipleBoxes = previouslySnappedBoxes.size() > 1;
    auto currentlySnappedToMultipleBoxes = m_currentlySnappedBoxes.size() > 1;
    
    if (wasSnappedToMultipleBoxes && !currentlySnappedToMultipleBoxes) {
        auto box = chooseBoxToResnapTo(previouslySnappedBoxes, snapOffsetsHorizontal, snapOffsetsVertical);
        snapPointChanged |= preserveCurrentTargetForAxis(ScrollEventAxis::Horizontal, box);
        snapPointChanged |= preserveCurrentTargetForAxis(ScrollEventAxis::Vertical, box);

        updateCurrentlySnappedBoxes();
        LOG_WITH_STREAM(ScrollSnap, stream << "ScrollSnapAnimatorState::resnapAfterLayout() - multiple boxes snapped; chose " << box << " (changed " << snapPointChanged << ") m_currentlySnappedBoxes " << m_currentlySnappedBoxes);
    }

    return snapPointChanged;
}

bool ScrollSnapAnimatorState::setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis axis, ScrollOffset scrollOffset, const ScrollExtents& scrollExtents, float pageScale)
{
    auto activeIndex = closestSnapPointForOffset(axis, scrollOffset, scrollExtents, pageScale);
    if (activeIndex == activeSnapIndexForAxis(axis))
        return false;

    setActiveSnapIndexForAxisInternal(axis, activeIndex);
    return true;
}

bool ScrollSnapAnimatorState::setNearestScrollSnapIndexForOffset(ScrollOffset scrollOffset, const ScrollExtents& scrollExtents, float pageScale)
{
    bool snapIndexChanged = false;
    snapIndexChanged |= setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis::Horizontal, scrollOffset, scrollExtents, pageScale);
    snapIndexChanged |= setNearestScrollSnapIndexForAxisAndOffsetInternal(ScrollEventAxis::Vertical, scrollOffset, scrollExtents, pageScale);

    updateCurrentlySnappedBoxes();

    return snapIndexChanged;
}

void ScrollSnapAnimatorState::transitionToUserInteractionState()
{
    teardownAnimationForState(ScrollSnapState::UserInteraction);
}

void ScrollSnapAnimatorState::transitionToDestinationReachedState()
{
    teardownAnimationForState(ScrollSnapState::DestinationReached);
}

void ScrollSnapAnimatorState::teardownAnimationForState(ScrollSnapState state)
{
    ASSERT(state == ScrollSnapState::UserInteraction || state == ScrollSnapState::DestinationReached);
    if (m_currentState == state)
        return;

    m_scrollController.stopAnimatedScroll();
    m_currentState = state;
}

std::pair<float, std::optional<unsigned>> ScrollSnapAnimatorState::targetOffsetForStartOffset(ScrollEventAxis axis, const ScrollExtents& scrollExtents, float startOffset, FloatPoint predictedOffset, float pageScale, float initialDelta) const
{
    auto minScrollOffset = (axis == ScrollEventAxis::Horizontal) ? scrollExtents.minimumScrollOffset().x() : scrollExtents.minimumScrollOffset().y();
    auto maxScrollOffset = (axis == ScrollEventAxis::Horizontal) ? scrollExtents.maximumScrollOffset().x() : scrollExtents.maximumScrollOffset().y();

    const auto& snapOffsets = m_snapOffsetsInfo.offsetsForAxis(axis);
    if (snapOffsets.isEmpty())
        return std::make_pair(clampTo<float>(axis == ScrollEventAxis::Horizontal ? predictedOffset.x() : predictedOffset.y(), minScrollOffset, maxScrollOffset), std::nullopt);

    LayoutPoint predictedLayoutOffset(predictedOffset.x() / pageScale, predictedOffset.y() / pageScale);
    auto [targetOffset, snapIndex] = m_snapOffsetsInfo.closestSnapOffset(axis, LayoutSize { scrollExtents.viewportSize }, predictedLayoutOffset, initialDelta, LayoutUnit(startOffset / pageScale));
    return std::make_pair(pageScale * clampTo<float>(float { targetOffset }, minScrollOffset, maxScrollOffset), snapIndex);
}

TextStream& operator<<(TextStream& ts, const ScrollSnapAnimatorState& state)
{
    ts << "ScrollSnapAnimatorState";
    ts.dumpProperty("snap offsets x", state.snapOffsetsForAxis(ScrollEventAxis::Horizontal));
    ts.dumpProperty("snap offsets y", state.snapOffsetsForAxis(ScrollEventAxis::Vertical));

    ts.dumpProperty("active snap index x", state.activeSnapIndexForAxis(ScrollEventAxis::Horizontal));
    ts.dumpProperty("active snap index y", state.activeSnapIndexForAxis(ScrollEventAxis::Vertical));

    return ts;
}

} // namespace WebCore
