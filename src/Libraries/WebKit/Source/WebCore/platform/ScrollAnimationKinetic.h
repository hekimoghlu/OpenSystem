/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#include "ScrollAnimation.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformWheelEvent;

class ScrollAnimationKinetic final : public ScrollAnimation {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimationKinetic);
private:
    class PerAxisData {
    public:
        PerAxisData(double lower, double upper, double initialOffset, double initialVelocity);

        double offset() { return m_offset; }
        double velocity() { return m_velocity; }

        bool animateScroll(Seconds timeDelta);

    private:
        double m_lower { 0 };
        double m_upper { 0 };

        double m_coef1 { 0 };
        double m_coef2 { 0 };

        Seconds m_elapsedTime;
        double m_offset { 0 };
        double m_velocity { 0 };
    };

public:
    ScrollAnimationKinetic(ScrollAnimationClient&);
    virtual ~ScrollAnimationKinetic();

    bool startAnimatedScrollWithInitialVelocity(const FloatPoint& initialOffset, const FloatSize& velocity, const FloatSize& previousVelocity, bool mayHScroll, bool mayVScroll);
    bool retargetActiveAnimation(const FloatPoint& newOffset) final;

    void appendToScrollHistory(const PlatformWheelEvent&);
    void clearScrollHistory();

    FloatSize computeVelocity();

    MonotonicTime startTime() { return m_startTime; }
    FloatSize initialVelocity() { return m_initialVelocity; }
    FloatPoint initialOffset() { return m_initialOffset; }

    FloatSize accumulateVelocityFromPreviousGesture(const MonotonicTime, const FloatPoint&, const FloatSize&);

private:
    void serviceAnimation(MonotonicTime) final;
    String debugDescription() const final;

    std::optional<PerAxisData> m_horizontalData;
    std::optional<PerAxisData> m_verticalData;

    Vector<PlatformWheelEvent> m_scrollHistory;
    FloatPoint m_initialOffset;
    FloatSize m_initialVelocity;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLL_ANIMATION(WebCore::ScrollAnimationKinetic, type() == WebCore::ScrollAnimation::Type::Kinetic)
