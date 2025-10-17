/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#include "RenderBlockFlow.h"

namespace WebCore {

class HTMLProgressElement;

class RenderProgress final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderProgress);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderProgress);
public:
    RenderProgress(HTMLElement&, RenderStyle&&);
    virtual ~RenderProgress();

    double position() const { return m_position; }
    double animationProgress() const;
    MonotonicTime animationStartTime() const { return m_animationStartTime; }

    bool isDeterminate() const;
    void updateFromElement() override;

    HTMLProgressElement* progressElement() const;

private:
    ASCIILiteral renderName() const override { return "RenderProgress"_s; }
    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;

    void animationTimerFired();
    void updateAnimationState();

    double m_position;
    MonotonicTime m_animationStartTime;
    bool m_animating { false };
    Timer m_animationTimer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderProgress, isRenderProgress())
