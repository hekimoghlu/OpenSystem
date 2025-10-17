/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

#include "RenderBoxModelObject.h"

namespace WebCore {

class HTMLElement;
class Position;

class RenderLineBreak final : public RenderBoxModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderLineBreak);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderLineBreak);
public:
    RenderLineBreak(HTMLElement&, RenderStyle&&);
    virtual ~RenderLineBreak();

    // FIXME: The lies here keep render tree dump based test results unchanged.
    ASCIILiteral renderName() const final { return isWBR() ? "RenderWordBreak"_s : "RenderBR"_s; }

    IntRect linesBoundingBox() const;

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint& accumulatedOffset) const final;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed = nullptr) const final;
#if PLATFORM(IOS_FAMILY)
    void collectSelectionGeometries(Vector<SelectionGeometry>&, unsigned startOffset = 0, unsigned endOffset = std::numeric_limits<unsigned>::max()) final;
#endif

    bool isBR() const { return !hasWBRLineBreakFlag(); }
    bool isWBR() const { return hasWBRLineBreakFlag(); }
    bool isLineBreakOpportunity() const { return isWBR(); }

private:
    void node() const = delete;

    bool canHaveChildren() const final { return false; }
    void paint(PaintInfo&, const LayoutPoint&) final { }

    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) final;
    int caretMinOffset() const final;
    int caretMaxOffset() const final;
    bool canBeSelectionLeaf() const final;

    LayoutUnit lineHeight(bool firstLine, LineDirectionMode, LinePositionMode) const final;
    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode) const final;

    LayoutUnit marginTop() const final { return 0; }
    LayoutUnit marginBottom() const final { return 0; }
    LayoutUnit marginLeft() const final { return 0; }
    LayoutUnit marginRight() const final { return 0; }
    LayoutUnit marginBefore(const WritingMode) const final { return 0; }
    LayoutUnit marginAfter(const WritingMode) const final { return 0; }
    LayoutUnit marginStart(const WritingMode) const final { return 0; }
    LayoutUnit marginEnd(const WritingMode) const final { return 0; }
    LayoutUnit offsetWidth() const final { return linesBoundingBox().width(); }
    LayoutUnit offsetHeight() const final { return linesBoundingBox().height(); }
    LayoutRect borderBoundingBox() const final { return LayoutRect(LayoutPoint(), linesBoundingBox().size()); }
    LayoutRect frameRectForStickyPositioning() const final { ASSERT_NOT_REACHED(); return { }; }
    RepaintRects localRectsForRepaint(RepaintOutlineBounds) const final { return { }; }

    void updateFromStyle() final;
    bool requiresLayer() const final { return false; }

    mutable std::optional<LayoutUnit> m_cachedLineHeight { };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderLineBreak, isRenderLineBreak())
