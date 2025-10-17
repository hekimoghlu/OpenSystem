/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

#include "ScrollAnimation.h"
#include <wtf/TZoneMalloc.h>

#if HAVE(RUBBER_BANDING)

namespace WebCore {

class ScrollAnimationRubberBand final: public ScrollAnimation {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimationRubberBand);
public:
    ScrollAnimationRubberBand(ScrollAnimationClient&);
    virtual ~ScrollAnimationRubberBand();

    // targetOffset is the scroll offset when the animation has finished (i.e. scrolled to an edge).
    bool startRubberBandAnimation(const FloatSize& initialVelocity, const FloatSize& initialOverscroll);

private:
    void updateScrollExtents() final;
    void serviceAnimation(MonotonicTime) final;
    bool retargetActiveAnimation(const FloatPoint&) final;
    ScrollClamping clamping() const final { return ScrollClamping::Unclamped; }
    String debugDescription() const final;

    bool animateScroll(MonotonicTime);

    FloatSize m_initialVelocity;
    FloatSize m_initialOverscroll;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLL_ANIMATION(WebCore::ScrollAnimationRubberBand, type() == WebCore::ScrollAnimation::Type::RubberBand)

#endif // HAVE(RUBBER_BANDING)
