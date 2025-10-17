/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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

#if ENABLE(VIDEO)

#include "FloatPoint.h"
#include "InlineIteratorInlineBox.h"
#include "RenderBlockFlow.h"

namespace WebCore {

class RenderBox;
class VTTCue;
class VTTCueBox;

class RenderVTTCue final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderVTTCue);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderVTTCue);
public:
    RenderVTTCue(VTTCueBox&, RenderStyle&&);
    virtual ~RenderVTTCue();

private:
    void layout() override;

    bool isOutside() const;
    bool rectIsWithinContainer(const IntRect&) const;
    bool isOverlapping() const;
    RenderVTTCue* overlappingObject() const;
    RenderVTTCue* overlappingObjectForRect(const IntRect&) const;
    bool shouldSwitchDirection(const InlineIterator::InlineBox&, LayoutUnit) const;

    void moveBoxesByStep(LayoutUnit);
    bool switchDirection(bool&, LayoutUnit&);
    void moveIfNecessaryToKeepWithinContainer();
    bool findNonOverlappingPosition(int& x, int& y) const;

    bool initializeLayoutParameters(LayoutUnit&, LayoutUnit&);
    void placeBoxInDefaultPosition(LayoutUnit, bool&);
    void repositionCueSnapToLinesSet();
    void repositionCueSnapToLinesNotSet();
    void repositionGenericCue();

    RenderBlockFlow& backdropBox() const;
    RenderInline* cueBox() const;

    VTTCue* m_cue;
    FloatPoint m_fallbackPosition;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderVTTCue, isRenderVTTCue())

#endif // ENABLE(VIDEO)
