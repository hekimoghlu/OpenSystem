/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#include "OpacityCaretAnimator.h"

#if HAVE(REDESIGNED_TEXT_CURSOR)

#include "FloatRoundedRect.h"
#include "VisibleSelection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpacityCaretAnimator);

static constexpr std::array keyframes = {
    KeyFrame { 0.0_s   , 1.00 },
    KeyFrame { 0.5_s   , 1.00 },
    KeyFrame { 0.5375_s, 0.75 },
    KeyFrame { 0.575_s , 0.50 },
    KeyFrame { 0.6125_s, 0.25 },
    KeyFrame { 0.65_s  , 0.00 },
    KeyFrame { 0.85_s  , 0.00 },
    KeyFrame { 0.8875_s, 0.25 },
    KeyFrame { 0.925_s , 0.50 },
    KeyFrame { 0.9625_s, 0.75 },
    KeyFrame { 1.0_s   , 1.00 },
};

OpacityCaretAnimator::OpacityCaretAnimator(CaretAnimationClient& client, std::optional<LayoutRect> repaintExpansionRect)
    : CaretAnimator(client)
    , m_overrideRepaintRect(repaintExpansionRect)
{
}

Seconds OpacityCaretAnimator::keyframeTimeDelta() const
{
    ASSERT(m_currentKeyframeIndex > 0 && m_currentKeyframeIndex < keyframes.size());
    return keyframes[m_currentKeyframeIndex].time - keyframes[m_currentKeyframeIndex - 1].time;
}

void OpacityCaretAnimator::setBlinkingSuspended(bool suspended)
{
    if (suspended == isBlinkingSuspended())
        return;

    if (!suspended) {
        m_currentKeyframeIndex = 1;
        m_blinkTimer.startOneShot(keyframeTimeDelta());
    }

    CaretAnimator::setBlinkingSuspended(suspended);
}

void OpacityCaretAnimator::updateAnimationProperties()
{
    if (isBlinkingSuspended() && m_presentationProperties.opacity >= 1.0)
        return;

    auto currentTime = MonotonicTime::now();
    if (currentTime - m_lastTimeCaretOpacityWasToggled >= keyframeTimeDelta()) {
        setOpacity(keyframes[m_currentKeyframeIndex].value);
        m_lastTimeCaretOpacityWasToggled = currentTime;

        if (m_currentKeyframeIndex == keyframes.size() - 1)
            m_currentKeyframeIndex = 0;

        m_currentKeyframeIndex++;

        m_blinkTimer.startOneShot(keyframeTimeDelta());

        m_overrideRepaintRect = std::nullopt;
    }
}

void OpacityCaretAnimator::start()
{
    // The default/start value of `m_currentKeyframeIndex` should be `1` since the keyframe
    // delta is the difference between `m_currentKeyframeIndex` and `m_currentKeyframeIndex - 1`
    m_currentKeyframeIndex = 1;
    m_lastTimeCaretOpacityWasToggled = MonotonicTime::now();
    didStart(m_lastTimeCaretOpacityWasToggled, keyframeTimeDelta());
}

String OpacityCaretAnimator::debugDescription() const
{
    TextStream textStream;
    textStream << "OpacityCaretAnimator " << this << " active " << isActive() << " opacity = " << m_presentationProperties.opacity;
    return textStream.release();
}

void OpacityCaretAnimator::paint(GraphicsContext& context, const FloatRect& rect, const Color& caretColor, const LayoutPoint&) const
{
    auto caretPresentationProperties = presentationProperties();

    auto caretColorWithOpacity = caretColor;
    if (caretColor != Color::transparentBlack)
        caretColorWithOpacity = caretColor.colorWithAlpha(caretPresentationProperties.opacity);

    context.fillRoundedRect(FloatRoundedRect { rect, FloatRoundedRect::Radii { 1.0 } }, caretColorWithOpacity);
}

LayoutRect OpacityCaretAnimator::caretRepaintRectForLocalRect(LayoutRect repaintRect) const
{
    if (!m_overrideRepaintRect)
        return CaretAnimator::caretRepaintRectForLocalRect(repaintRect);

    auto rect = *m_overrideRepaintRect;
    repaintRect.moveBy(rect.location());
    repaintRect.expand(rect.size());

    return repaintRect;
}

} // namespace WebCore

#endif
