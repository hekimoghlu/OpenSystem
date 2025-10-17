/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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
#include "LayoutIntegrationFlexLayout.h"

#include "FlexFormattingConstraints.h"
#include "FlexFormattingContext.h"
#include "FormattingContextBoxIterator.h"
#include "HitTestLocation.h"
#include "HitTestRequest.h"
#include "HitTestResult.h"
#include "LayoutBoxGeometry.h"
#include "LayoutChildIterator.h"
#include "LayoutIntegrationBoxGeometryUpdater.h"
#include "RenderBox.h"
#include "RenderBoxInlines.h"
#include "RenderFlexibleBox.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace LayoutIntegration {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FlexLayout);

FlexLayout::FlexLayout(RenderFlexibleBox& flexBoxRenderer)
    : m_flexBox(BoxTreeUpdater { flexBoxRenderer }.build())
    , m_layoutState(flexBoxRenderer.view().layoutState())
{
}

FlexLayout::~FlexLayout()
{
    auto& renderer = flexBoxRenderer();
    m_flexBox = nullptr;

    BoxTreeUpdater { renderer }.tearDown();
}

static inline Layout::ConstraintsForFlexContent constraintsForFlexContent(const Layout::ElementBox& flexContainer)
{
    auto& flexContainerRenderer = downcast<RenderFlexibleBox>(*flexContainer.rendererForIntegration());
    auto& flexBoxStyle = flexContainer.style();
    auto boxSizingIsContentBox = flexBoxStyle.boxSizing() == BoxSizing::ContentBox;
    auto availableLogicalWidth = flexContainerRenderer.contentBoxRect().width();
    // FIXME: Use root's BoxGeometry which first needs to stop flipping for the formatting context.
    auto horizontallMarginBorderAndPadding = flexContainerRenderer.marginAndBorderAndPaddingStart() + flexContainerRenderer.marginAndBorderAndPaddingEnd();
    auto verticalMarginBorderAndPadding = flexContainerRenderer.marginAndBorderAndPaddingBefore() + flexContainerRenderer.marginAndBorderAndPaddingAfter();

    auto widthValue = [&](auto& computedValue) -> std::optional<LayoutUnit> {
        if (computedValue.isFixed())
            return LayoutUnit { boxSizingIsContentBox ? computedValue.value() : computedValue.value() - horizontallMarginBorderAndPadding };

        if (computedValue.isPercent()) {
            auto value = valueForLength(computedValue, flexContainerRenderer.containingBlock()->logicalWidth());
            return LayoutUnit { boxSizingIsContentBox ? value : value - horizontallMarginBorderAndPadding };
        }
        return { };
    };

    auto heightValue = [&](auto& computedValue, bool callRendererForPercentValue = false) -> std::optional<LayoutUnit> {
        if (computedValue.isFixed())
            return LayoutUnit { boxSizingIsContentBox ? computedValue.value() : computedValue.value() - verticalMarginBorderAndPadding };

        if (computedValue.isPercent()) {
            if (callRendererForPercentValue)
                return flexContainerRenderer.computePercentageLogicalHeight(computedValue, RenderBox::UpdatePercentageHeightDescendants::No);

            if (flexContainerRenderer.containingBlock()->style().logicalHeight().isFixed()) {
                auto value = valueForLength(computedValue, flexContainerRenderer.containingBlock()->style().height().value());
                return LayoutUnit { boxSizingIsContentBox ? value : value - verticalMarginBorderAndPadding };
            }
        }
        return { };
    };

    auto widthGeometry = [&]() -> Layout::ConstraintsForFlexContent::AxisGeometry {
        return { widthValue(flexBoxStyle.minWidth()), widthValue(flexBoxStyle.maxWidth()), availableLogicalWidth ? availableLogicalWidth : widthValue(flexBoxStyle.width()), flexContainerRenderer.contentBoxLocation().x() };
    };

    auto heightGeometry = [&]() -> Layout::ConstraintsForFlexContent::AxisGeometry {
        auto availableSize = heightValue(flexBoxStyle.height(), true);
        auto logicalMinHeight = heightValue(flexBoxStyle.minHeight()).value_or(0_lu);
        auto logicalMaxHeight = heightValue(flexBoxStyle.maxHeight());
        if (!availableSize || (logicalMaxHeight && *logicalMaxHeight < *availableSize))
            availableSize = logicalMaxHeight;

        return Layout::ConstraintsForFlexContent::AxisGeometry { logicalMinHeight, logicalMaxHeight, availableSize, flexContainerRenderer.contentBoxLocation().y() };
    };

    return Layout::FlexFormattingUtils::isMainAxisParallelWithInlineAxis(flexContainer) ? Layout::ConstraintsForFlexContent(widthGeometry(), heightGeometry(), false) : Layout::ConstraintsForFlexContent(heightGeometry(), widthGeometry(), false);
}

void FlexLayout::updateFormattingContexGeometries()
{
    auto boxGeometryUpdater = BoxGeometryUpdater { layoutState(), flexBox() };
    boxGeometryUpdater.setFormattingContextRootGeometry(flexBoxRenderer().containingBlock()->contentBoxLogicalWidth());
    boxGeometryUpdater.setFormattingContextContentGeometry(layoutState().geometryForBox(flexBox()).contentBoxWidth(), { });
}

void FlexLayout::updateStyle(const RenderBlock&, const RenderStyle&)
{
}

std::pair<LayoutUnit, LayoutUnit> FlexLayout::computeIntrinsicWidthConstraints()
{
    auto flexFormattingContext = Layout::FlexFormattingContext { flexBox(), layoutState() };
    auto constraints = flexFormattingContext.computedIntrinsicWidthConstraints();

    return { constraints.minimum, constraints.maximum };
}

void FlexLayout::layout()
{
    Layout::FlexFormattingContext { flexBox(), layoutState() }.layout(constraintsForFlexContent(flexBox()));

    updateRenderers();

    auto relayoutFlexItems = [&] {
        // Flex items need to be laid out now with their final size (and through setOverridingBorderBoxLogicalWidth/Height)
        // Note that they may re-size themselves.
        auto flexContainerIsHorizontal = flexBox().writingMode().isHorizontal();
        for (auto& layoutBox : formattingContextBoxes(flexBox())) {
            auto& renderer = downcast<RenderBox>(*layoutBox.rendererForIntegration());
            auto isOrthogonal = flexContainerIsHorizontal != renderer.writingMode().isHorizontal();
            auto borderBox = Layout::BoxGeometry::borderBoxRect(layoutState().geometryForBox(layoutBox));

            renderer.setWidth(LayoutUnit { });
            renderer.setHeight(LayoutUnit { });
            // logical here means width and height constraints for the _content_ of the flex items not the flex items' own dimension inside the flex container.
            renderer.setOverridingBorderBoxLogicalWidth(isOrthogonal ? borderBox.height() : borderBox.width());
            renderer.setOverridingBorderBoxLogicalHeight(isOrthogonal ? borderBox.width() : borderBox.height());

            renderer.setChildNeedsLayout(MarkOnlyThis);
            renderer.layoutIfNeeded();
            renderer.clearOverridingSize();

            renderer.setWidth(flexContainerIsHorizontal ? borderBox.width() : borderBox.height());
            renderer.setHeight(flexContainerIsHorizontal ? borderBox.height() : borderBox.width());
        }
    };
    relayoutFlexItems();
}

void FlexLayout::updateRenderers()
{
    auto flexContainerIsHorizontal = flexBox().writingMode().isHorizontal();
    for (auto& layoutBox : formattingContextBoxes(flexBox())) {
        auto& renderer = downcast<RenderBox>(*layoutBox.rendererForIntegration());
        auto& flexItemGeometry = layoutState().geometryForBox(layoutBox);
        auto borderBox = Layout::BoxGeometry::borderBoxRect(flexItemGeometry);
        renderer.setLocation(flexContainerIsHorizontal ? borderBox.topLeft() : borderBox.topLeft().transposedPoint());
        renderer.setWidth(flexContainerIsHorizontal ? borderBox.width() : borderBox.height());
        renderer.setHeight(flexContainerIsHorizontal ? borderBox.height() : borderBox.width());

        renderer.setMarginStart(flexItemGeometry.marginStart());
        renderer.setMarginEnd(flexItemGeometry.marginEnd());
        renderer.setMarginBefore(flexItemGeometry.marginBefore());
        renderer.setMarginAfter(flexItemGeometry.marginAfter());

        if (!renderer.everHadLayout() || renderer.checkForRepaintDuringLayout())
            renderer.repaint();
    }
}

void FlexLayout::paint(PaintInfo&, const LayoutPoint&)
{
}

bool FlexLayout::hitTest(const HitTestRequest&, HitTestResult&, const HitTestLocation&, const LayoutPoint&, HitTestAction)
{
    return false;
}

void FlexLayout::collectOverflow()
{
}

LayoutUnit FlexLayout::contentBoxLogicalHeight() const
{
    auto contentLogicalBottom = LayoutUnit { };
    for (auto& layoutBox : formattingContextBoxes(flexBox()))
        contentLogicalBottom = std::max(contentLogicalBottom, Layout::BoxGeometry::marginBoxRect(layoutState().geometryForBox(layoutBox)).bottom());
    return contentLogicalBottom - layoutState().geometryForBox(flexBox()).contentBoxTop();
}

}
}

