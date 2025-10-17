/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

namespace WebCore {

class SimpleCaretAnimator final : public CaretAnimator {
    WTF_MAKE_TZONE_ALLOCATED(SimpleCaretAnimator);
public:
    explicit SimpleCaretAnimator(CaretAnimationClient&);

private:
    void updateAnimationProperties() final;
    void start() final;

    String debugDescription() const final;

    void setVisible(bool visible) final { setBlinkState(visible ? PresentationProperties::BlinkState::On : PresentationProperties::BlinkState::Off); }

    void setBlinkState(PresentationProperties::BlinkState blinkState)
    { 
        if (m_presentationProperties.blinkState == blinkState)
            return;

        m_presentationProperties.blinkState = blinkState; 
        m_client.caretAnimationDidUpdate(*this);
    }

    MonotonicTime m_lastTimeCaretPaintWasToggled;
};

} // namespace WebCore
