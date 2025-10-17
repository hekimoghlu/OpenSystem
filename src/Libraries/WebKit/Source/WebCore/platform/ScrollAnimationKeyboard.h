/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#include "KeyboardScroll.h"
#include "RectEdges.h"
#include "ScrollAnimation.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPoint;
class TimingFunction;

class ScrollAnimationKeyboard final: public ScrollAnimation {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimationKeyboard);
public:
    ScrollAnimationKeyboard(ScrollAnimationClient&);
    virtual ~ScrollAnimationKeyboard();

    bool startKeyboardScroll(const KeyboardScroll&);

    void finishKeyboardScroll(bool immediate);

    void stopKeyboardScrollAnimation();

    ScrollClamping clamping() const override { return ScrollClamping::Unclamped; }

private:
    void serviceAnimation(MonotonicTime) final;
    bool retargetActiveAnimation(const FloatPoint&) final;

    String debugDescription() const final;

    bool animateScroll(MonotonicTime);

    RectEdges<bool> scrollableDirectionsFromPosition(FloatPoint position);

    std::optional<KeyboardScroll> m_currentKeyboardScroll;
    FloatSize m_velocity;
    MonotonicTime m_timeAtLastFrame;
    FloatPoint m_idealPosition;
    FloatSize m_idealPositionForMinimumTravel;
    bool m_scrollTriggeringKeyIsPressed;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLL_ANIMATION(WebCore::ScrollAnimationKeyboard, type() == WebCore::ScrollAnimation::Type::Keyboard)
