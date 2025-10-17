/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#include "FormattingConstraints.h"
#include "InlineFormattingConstraints.h"
#include "LayoutBoxGeometry.h"
#include "LayoutIntegrationBoxTreeUpdater.h"
#include "LayoutState.h"

namespace WebCore {

class RenderBox;
class RenderBlock;
class RenderLineBreak;
class RenderInline;
class RenderTable;
class RenderListItem;
class RenderListMarker;

namespace LayoutIntegration {

class BoxGeometryUpdater {
public:
    BoxGeometryUpdater(Layout::LayoutState&, const Layout::ElementBox& rootLayoutBox);

    void clear();

    void setFormattingContextRootGeometry(LayoutUnit availableLogicalWidth);
    void setFormattingContextContentGeometry(std::optional<LayoutUnit> availableLogicalWidth, std::optional<Layout::IntrinsicWidthMode>);
    void updateBoxGeometryAfterIntegrationLayout(const Layout::ElementBox&, LayoutUnit availableWidth);

    Layout::ConstraintsForInlineContent formattingContextConstraints(LayoutUnit availableWidth);

    UncheckedKeyHashMap<const Layout::ElementBox*, LayoutUnit> takeNestedListMarkerOffsets() { return WTFMove(m_nestedListMarkerOffsets); }

private:
    void updateBoxGeometry(const RenderElement&, std::optional<LayoutUnit> availableWidth, std::optional<Layout::IntrinsicWidthMode>);

    void updateLayoutBoxDimensions(const RenderBox&, std::optional<LayoutUnit> availableWidth, std::optional<Layout::IntrinsicWidthMode> = std::nullopt);
    void updateLineBreakBoxDimensions(const RenderLineBreak&);
    void updateInlineBoxDimensions(const RenderInline&, std::optional<LayoutUnit> availableWidth, std::optional<Layout::IntrinsicWidthMode> = std::nullopt);
    void setListMarkerOffsetForMarkerOutside(const RenderListMarker&);

    Layout::BoxGeometry::HorizontalEdges horizontalLogicalMargin(const RenderBoxModelObject&, std::optional<LayoutUnit> availableWidth, WritingMode, bool retainMarginStart = true, bool retainMarginEnd = true);
    Layout::BoxGeometry::VerticalEdges verticalLogicalMargin(const RenderBoxModelObject&, std::optional<LayoutUnit> availableWidth, WritingMode);
    Layout::BoxGeometry::Edges logicalBorder(const RenderBoxModelObject&, WritingMode, bool isIntrinsicWidthMode = false, bool retainBorderStart = true, bool retainBorderEnd = true);
    Layout::BoxGeometry::Edges logicalPadding(const RenderBoxModelObject&, std::optional<LayoutUnit> availableWidth, WritingMode, bool retainPaddingStart = true, bool retainPaddingEnd = true);

    Layout::LayoutState& layoutState() { return *m_layoutState; }
    const Layout::LayoutState& layoutState() const { return *m_layoutState; }
    const Layout::ElementBox& rootLayoutBox() const;
    const RenderBlock& rootRenderer() const;
    inline WritingMode writingMode() const;

private:
    WeakPtr<Layout::LayoutState> m_layoutState;
    CheckedPtr<const Layout::ElementBox> m_rootLayoutBox;
    UncheckedKeyHashMap<const Layout::ElementBox*, LayoutUnit> m_nestedListMarkerOffsets;
};

}
}

