/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "LayoutIntegrationFormattingContextLayout.h"

#include "LayoutIntegrationBoxGeometryUpdater.h"
#include "RenderBlock.h"
#include "RenderBoxInlines.h"
#include "RenderFlexibleBox.h"

namespace WebCore {
namespace LayoutIntegration {

void layoutWithFormattingContextForBox(const Layout::ElementBox& box, std::optional<LayoutUnit> widthConstraint, Layout::LayoutState& layoutState)
{
    auto& renderer = downcast<RenderBox>(*box.rendererForIntegration());

    if (widthConstraint) {
        renderer.setOverridingBorderBoxLogicalWidth(*widthConstraint);
        renderer.setNeedsLayout(MarkOnlyThis);
    }

    renderer.layoutIfNeeded();

    if (widthConstraint)
        renderer.clearOverridingBorderBoxLogicalWidth();

    auto rootLayoutBox = [&]() -> const Layout::ElementBox& {
        auto* ancestor = &box.parent();
        while (!ancestor->isInitialContainingBlock()) {
            if (ancestor->establishesFormattingContext())
                break;
            ancestor = &ancestor->parent();
        }
        return *ancestor;
    };
    auto updater = BoxGeometryUpdater { layoutState, rootLayoutBox() };
    updater.updateBoxGeometryAfterIntegrationLayout(box, widthConstraint.value_or(renderer.containingBlock()->contentBoxLogicalWidth()));
}

LayoutUnit formattingContextRootLogicalWidthForType(const Layout::ElementBox& box, LogicalWidthType logicalWidthType)
{
    ASSERT(box.establishesFormattingContext());

    auto& renderer = downcast<RenderBox>(*box.rendererForIntegration());
    switch (logicalWidthType) {
    case LogicalWidthType::PreferredMaximum:
        return renderer.maxPreferredLogicalWidth();
    case LogicalWidthType::PreferredMinimum:
        return renderer.minPreferredLogicalWidth();
    case LogicalWidthType::MaxContent:
    case LogicalWidthType::MinContent: {
        auto minimunLogicalWidth = LayoutUnit { };
        auto maximumLogicalWidth = LayoutUnit { };
        renderer.computeIntrinsicLogicalWidths(minimunLogicalWidth, maximumLogicalWidth);
        return logicalWidthType == LogicalWidthType::MaxContent ? maximumLogicalWidth : minimunLogicalWidth;
    }
    default:
        ASSERT_NOT_REACHED();
        return { };
    }
}

LayoutUnit formattingContextRootLogicalHeightForType(const Layout::ElementBox& box, LogicalHeightType logicalHeightType)
{
    ASSERT(box.establishesFormattingContext());

    auto& renderer = downcast<RenderBox>(*box.rendererForIntegration());
    switch (logicalHeightType) {
    case LogicalHeightType::MinContent: {
        // Since currently we can't ask RenderBox for content height, this is limited to flex items
        // where the legacy flex layout "fixed" this by caching the content height in RenderBox::updateLogicalHeight
        // before additional height constraints applied.
        if (auto* flexContainer = dynamicDowncast<RenderFlexibleBox>(renderer.parent()))
            return flexContainer->cachedFlexItemIntrinsicContentLogicalHeight(renderer);
        ASSERT_NOT_IMPLEMENTED_YET();
        return { };
    }
    default:
        ASSERT_NOT_REACHED();
        return { };
    }
}

}
}
