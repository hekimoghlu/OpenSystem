/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#include "RenderBlock.h"

namespace WebCore {

class FlexBoxIterator;

class RenderDeprecatedFlexibleBox final : public RenderBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderDeprecatedFlexibleBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderDeprecatedFlexibleBox);
public:
    RenderDeprecatedFlexibleBox(Element&, RenderStyle&&);
    virtual ~RenderDeprecatedFlexibleBox();

    Element& element() const { return downcast<Element>(nodeForNonAnonymous()); }

    ASCIILiteral renderName() const override;

    void styleWillChange(StyleDifference, const RenderStyle& newStyle) override;

    void layoutBlock(bool relayoutChildren, LayoutUnit pageHeight = 0_lu) override;
    void layoutHorizontalBox(bool relayoutChildren);
    void layoutVerticalBox(bool relayoutChildren);

    bool isStretchingChildren() const { return m_stretchingChildren; }

    bool canDropAnonymousBlockChild() const override { return false; }

private:
    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;
    void computePreferredLogicalWidths() override;

    LayoutUnit allowedChildFlex(RenderBox* child, bool expanding, unsigned group);
    void placeChild(RenderBox* child, const LayoutPoint& location, LayoutSize* childLayoutDelta = nullptr);

    bool hasMultipleLines() const { return style().boxLines() == BoxLines::Multiple; }
    bool isVertical() const { return style().boxOrient() == BoxOrient::Vertical; }
    bool isHorizontal() const { return style().boxOrient() == BoxOrient::Horizontal; }

    struct ClampedContent {
        LayoutUnit contentHeight;
        SingleThreadWeakPtr<const RenderBlockFlow> renderer;
    };
    ClampedContent applyLineClamp(FlexBoxIterator&, bool relayoutChildren);
    void clearLineClamp();

    bool m_stretchingChildren;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderDeprecatedFlexibleBox, isRenderDeprecatedFlexibleBox())
