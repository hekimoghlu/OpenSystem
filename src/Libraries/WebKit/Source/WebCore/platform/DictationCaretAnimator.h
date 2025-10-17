/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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

#include "CaretAnimator.h"
#include <wtf/TZoneMalloc.h>

#if HAVE(REDESIGNED_TEXT_CURSOR)

namespace WebCore {

class Path;

class DictationCaretAnimator final : public CaretAnimator {
    WTF_MAKE_TZONE_ALLOCATED(DictationCaretAnimator);
public:
    explicit DictationCaretAnimator(CaretAnimationClient&);

private:
    void updateAnimationProperties() final;
    void start() final;
    FloatRect tailRect() const;

    String debugDescription() const final;

    void setVisible(bool visible) final { setOpacity(visible ? 1.0 : 0.0); }

    void setOpacity(float opacity)
    {
        if (m_presentationProperties.opacity == opacity)
            return;

        m_presentationProperties.opacity = opacity;
        m_client.caretAnimationDidUpdate(*this);
    }

    void setBlinkingSuspended(bool) final;

    void stop(CaretAnimatorStopReason) final;

    Seconds keyframeTimeDelta() const;
    size_t keyframeCount() const;
    void updateGlowTail(float caretPosition, Seconds elapsedTime);
    void resetGlowTail(FloatRect);
    void updateGlowTail(Seconds elapsedTime);
    void paint(GraphicsContext&, const FloatRect&, const Color&, const LayoutPoint&) const final;
    LayoutRect caretRepaintRectForLocalRect(LayoutRect repaintRect) const final;
    Path makeDictationTailConePath(const FloatRect&) const;
    void fillCaretTail(const FloatRect&, GraphicsContext&, const Color&) const;

    FloatRect computeTailRect() const;
    bool isLeftToRightLayout() const;
    FloatRoundedRect expandedCaretRect(const FloatRect&, bool fillTail) const;
    int computeScrollLeft() const;

    MonotonicTime m_lastUpdateTime;
    size_t m_currentKeyframeIndex { 1 };
    FloatRect m_localCaretRect;
    FloatRect m_tailRect, m_previousTailRect;
    float m_animationSpeed { 0 };
    float m_glowStart { 0 };
    float m_initialScale { 0 };
    float m_scrollLeft { 0 };
};

} // namespace WebCore

#endif
