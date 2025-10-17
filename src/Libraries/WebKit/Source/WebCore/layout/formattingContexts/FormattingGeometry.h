/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "FormattingContext.h"
#include "LayoutBoxGeometry.h"

namespace WebCore {
namespace Layout {

struct ComputedHorizontalMargin;
struct ComputedVerticalMargin;
class ElementBox;
struct ContentHeightAndMargin;
struct ContentWidthAndMargin;
struct HorizontalGeometry;
class LayoutState;
struct OverriddenHorizontalValues;
struct OverriddenVerticalValues;
struct VerticalGeometry;

// This class implements generic positioning and sizing.
class FormattingGeometry {
public:
    FormattingGeometry(const FormattingContext&);

    VerticalGeometry outOfFlowVerticalGeometry(const Box&, const HorizontalConstraints&, const VerticalConstraints&, const OverriddenVerticalValues&) const;
    HorizontalGeometry outOfFlowHorizontalGeometry(const Box&, const HorizontalConstraints&, const VerticalConstraints&, const OverriddenHorizontalValues&) const;

    ContentHeightAndMargin floatingContentHeightAndMargin(const Box&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin floatingContentWidthAndMargin(const Box&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    ContentHeightAndMargin inlineReplacedContentHeightAndMargin(const ElementBox&, const HorizontalConstraints&, std::optional<VerticalConstraints>, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin inlineReplacedContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, std::optional<VerticalConstraints>, const OverriddenHorizontalValues&) const;

    LayoutSize inFlowPositionedPositionOffset(const Box&, const HorizontalConstraints&) const;

    ContentHeightAndMargin complicatedCases(const Box&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    LayoutUnit shrinkToFitWidth(const Box&, LayoutUnit availableWidth) const;

    BoxGeometry::Edges computedBorder(const Box&) const;
    BoxGeometry::Edges computedPadding(const Box&, LayoutUnit containingBlockWidth) const;

    ComputedHorizontalMargin computedHorizontalMargin(const Box&, const HorizontalConstraints&) const;
    ComputedVerticalMargin computedVerticalMargin(const Box&, const HorizontalConstraints&) const;

    std::optional<LayoutUnit> computedValue(const Length& geometryProperty, LayoutUnit containingBlockWidth) const;
    std::optional<LayoutUnit> fixedValue(const Length& geometryProperty) const;

    std::optional<LayoutUnit> computedMinHeight(const Box&, std::optional<LayoutUnit> containingBlockHeight = std::nullopt) const;
    std::optional<LayoutUnit> computedMaxHeight(const Box&, std::optional<LayoutUnit> containingBlockHeight = std::nullopt) const;

    std::optional<LayoutUnit> computedMinWidth(const Box&, LayoutUnit containingBlockWidth) const;
    std::optional<LayoutUnit> computedMaxWidth(const Box&, LayoutUnit containingBlockWidth) const;

    IntrinsicWidthConstraints constrainByMinMaxWidth(const Box&, IntrinsicWidthConstraints) const;

    LayoutUnit contentHeightForFormattingContextRoot(const ElementBox&) const;

    ConstraintsForOutOfFlowContent constraintsForOutOfFlowContent(const ElementBox&) const;
    ConstraintsForInFlowContent constraintsForInFlowContent(const ElementBox&, std::optional<FormattingContext::EscapeReason> = std::nullopt) const;

    std::optional<LayoutUnit> computedHeight(const Box&, std::optional<LayoutUnit> containingBlockHeight = std::nullopt) const;
    std::optional<LayoutUnit> computedWidth(const Box&, LayoutUnit containingBlockWidth) const;

    bool isBlockFormattingGeometry() const { return formattingContext().isBlockFormattingContext(); }
    bool isTableFormattingGeometry() const { return formattingContext().isTableFormattingContext(); }

protected:
    const LayoutState& layoutState() const { return m_formattingContext.layoutState(); }
    const FormattingContext& formattingContext() const { return m_formattingContext; }

private:
    VerticalGeometry outOfFlowReplacedVerticalGeometry(const ElementBox&, const HorizontalConstraints&, const VerticalConstraints&, const OverriddenVerticalValues&) const;
    HorizontalGeometry outOfFlowReplacedHorizontalGeometry(const ElementBox&, const HorizontalConstraints&, const VerticalConstraints&, const OverriddenHorizontalValues&) const;

    VerticalGeometry outOfFlowNonReplacedVerticalGeometry(const ElementBox&, const HorizontalConstraints&, const VerticalConstraints&, const OverriddenVerticalValues&) const;
    HorizontalGeometry outOfFlowNonReplacedHorizontalGeometry(const ElementBox&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    ContentHeightAndMargin floatingReplacedContentHeightAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin floatingReplacedContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    ContentWidthAndMargin floatingNonReplacedContentWidthAndMargin(const Box&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    LayoutUnit staticVerticalPositionForOutOfFlowPositioned(const Box&, const VerticalConstraints&) const;
    LayoutUnit staticHorizontalPositionForOutOfFlowPositioned(const Box&, const HorizontalConstraints&) const;

    enum class HeightType { Min, Max, Normal };
    std::optional<LayoutUnit> computedHeightValue(const Box&, HeightType, std::optional<LayoutUnit> containingBlockHeight) const;

    enum class WidthType { Min, Max, Normal };
    std::optional<LayoutUnit> computedWidthValue(const Box&, WidthType, LayoutUnit containingBlockWidth) const;

    const FormattingContext& m_formattingContext;
};

}
}

#define SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_GEOMETRY(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Layout::ToValueTypeName) \
    static bool isType(const WebCore::Layout::FormattingGeometry& formattingGeometry) { return formattingGeometry.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

