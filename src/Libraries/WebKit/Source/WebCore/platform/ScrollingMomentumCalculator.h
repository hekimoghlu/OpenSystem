/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "PlatformWheelEvent.h"
#include "ScrollExtents.h"
#include "ScrollTypes.h"
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPoint;
class FloatSize;

class ScrollingMomentumCalculator {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingMomentumCalculator);
public:
    WEBCORE_EXPORT static void setPlatformMomentumScrollingPredictionEnabled(bool);

    static std::unique_ptr<ScrollingMomentumCalculator> create(const ScrollExtents&, const FloatPoint& initialOffset, const FloatSize& initialDelta, const FloatSize& initialVelocity);

    ScrollingMomentumCalculator(const ScrollExtents&, const FloatPoint& initialOffset, const FloatSize& initialDelta, const FloatSize& initialVelocity);
    virtual ~ScrollingMomentumCalculator() = default;

    virtual FloatPoint scrollOffsetAfterElapsedTime(Seconds) = 0;
    virtual Seconds animationDuration() = 0;

    FloatPoint destinationScrollOffset() const { return m_retargetedScrollOffset.value_or(m_initialDestinationOffset); }

    void setRetargetedScrollOffset(const FloatPoint&);

protected:
    virtual FloatPoint predictedDestinationOffset();

    virtual void destinationScrollOffsetDidChange() { }

    FloatSize m_initialDelta;
    FloatSize m_initialVelocity;
    FloatPoint m_initialScrollOffset;
    FloatPoint m_initialDestinationOffset;
    ScrollExtents m_scrollExtents;

private:
    std::optional<FloatPoint> m_retargetedScrollOffset;
};

class BasicScrollingMomentumCalculator final : public ScrollingMomentumCalculator {
public:
    BasicScrollingMomentumCalculator(const ScrollExtents&, const FloatPoint& initialOffset, const FloatSize& initialDelta, const FloatSize& initialVelocity);

private:
    FloatPoint scrollOffsetAfterElapsedTime(Seconds) final;
    Seconds animationDuration() final;

    void initializeInterpolationCoefficientsIfNecessary();
    void initializeSnapProgressCurve();
    float animationProgressAfterElapsedTime(Seconds) const;

    FloatPoint linearlyInterpolatedOffsetAtProgress(float progress);
    FloatPoint cubicallyInterpolatedOffsetAtProgress(float progress) const;

    float m_snapAnimationCurveMagnitude { 0 };
    float m_snapAnimationDecayFactor { 0 };
    std::array<FloatSize, 4> m_snapAnimationCurveCoefficients = { };
    bool m_forceLinearAnimationCurve { false };
    bool m_momentumCalculatorRequiresInitialization { true };
};

} // namespace WebCore
