/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#include "FlexFormattingConstraints.h"
#include "FlexFormattingUtils.h"
#include "FlexLayout.h"
#include "FlexRect.h"
#include "LayoutIntegrationUtils.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

// This class implements the layout logic for flex formatting contexts.
// https://www.w3.org/TR/css-flexbox-1/
class FlexFormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FlexFormattingContext);
public:
    FlexFormattingContext(const ElementBox& flexBox, LayoutState&);

    void layout(const ConstraintsForFlexContent&);
    IntrinsicWidthConstraints computedIntrinsicWidthConstraints();

    const ElementBox& root() const { return m_flexBox; }
    const FlexFormattingUtils& formattingUtils() const { return m_flexFormattingUtils; }

    const BoxGeometry& geometryForFlexItem(const Box&) const;
    BoxGeometry& geometryForFlexItem(const Box&);

    const IntegrationUtils& integrationUtils() const { return m_integrationUtils; }

private:
    FlexLayout::LogicalFlexItems convertFlexItemsToLogicalSpace(const ConstraintsForFlexContent&);
    void setFlexItemsGeometry(const FlexLayout::LogicalFlexItems&, const FlexLayout::LogicalFlexItemRects&, const ConstraintsForFlexContent&);
    void positionOutOfFlowChildren();

    std::optional<LayoutUnit> computedAutoMarginValueForFlexItems(const ConstraintsForFlexContent&);

private:
    const ElementBox& m_flexBox;
    LayoutState& m_globalLayoutState;
    const FlexFormattingUtils m_flexFormattingUtils;
    const IntegrationUtils m_integrationUtils;
};

}
}

