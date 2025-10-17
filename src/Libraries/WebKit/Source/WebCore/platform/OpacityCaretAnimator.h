/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

class OpacityCaretAnimator final : public CaretAnimator {
    WTF_MAKE_TZONE_ALLOCATED(OpacityCaretAnimator);
public:
    explicit OpacityCaretAnimator(CaretAnimationClient&, std::optional<LayoutRect> = std::nullopt);

private:
    void updateAnimationProperties() final;
    void start() final;
    void paint(GraphicsContext&, const FloatRect&, const Color&, const LayoutPoint&) const final;

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

    Seconds keyframeTimeDelta() const;
    LayoutRect caretRepaintRectForLocalRect(LayoutRect) const final;

    MonotonicTime m_lastTimeCaretOpacityWasToggled;
    size_t m_currentKeyframeIndex { 1 };
    std::optional<LayoutRect> m_overrideRepaintRect;
};

} // namespace WebCore

#endif
