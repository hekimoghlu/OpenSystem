/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#include "SimpleCaretAnimator.h"

#include "RenderTheme.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SimpleCaretAnimator);

SimpleCaretAnimator::SimpleCaretAnimator(CaretAnimationClient& client)
    : CaretAnimator(client)
{
}

void SimpleCaretAnimator::updateAnimationProperties()
{
    auto currentTime = MonotonicTime::now();
    auto caretBlinkInterval = RenderTheme::singleton().caretBlinkInterval();

    setBlinkingSuspended(!caretBlinkInterval);

    // Ensure the caret is always visible when blinking is suspended.
    if (isBlinkingSuspended() && m_presentationProperties.blinkState == PresentationProperties::BlinkState::On) {
        m_blinkTimer.startOneShot(caretBlinkInterval.value_or(0_ms));
        return;
    }

    // If blinking is disabled, set isBlinkingSuspended() would have made the
    // previous check return early and at this point there must be an interval.
    ASSERT(caretBlinkInterval.has_value());

    if (currentTime - m_lastTimeCaretPaintWasToggled >= caretBlinkInterval) {
        setBlinkState(!m_presentationProperties.blinkState);
        m_lastTimeCaretPaintWasToggled = currentTime;

        m_blinkTimer.startOneShot(*caretBlinkInterval);
    }
}

void SimpleCaretAnimator::start()
{
    m_lastTimeCaretPaintWasToggled = MonotonicTime::now();
    didStart(m_lastTimeCaretPaintWasToggled, RenderTheme::singleton().caretBlinkInterval());
}

String SimpleCaretAnimator::debugDescription() const
{
    TextStream textStream;
    textStream << "SimpleCaretAnimator " << this << " active " << isActive() << " blink state = " << (m_presentationProperties.blinkState == PresentationProperties::BlinkState::On ? "On" : "Off");
    return textStream.release();
}

} // namespace WebCore
