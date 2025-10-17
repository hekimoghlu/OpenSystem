/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

#include "LayoutElementBox.h"

namespace WebCore {
namespace Layout {

class LogicalFlexItem {
public:
    struct MainAxisGeometry {
        LayoutUnit margin() const { return marginStart.value_or(0_lu) + marginEnd.value_or(0_lu); }

        std::optional<LayoutUnit> definiteFlexBasis;

        std::optional<LayoutUnit> size;
        std::optional<LayoutUnit> maximumSize;
        std::optional<LayoutUnit> minimumSize;

        std::optional<LayoutUnit> marginStart;
        std::optional<LayoutUnit> marginEnd;

        LayoutUnit borderAndPadding;
    };

    struct CrossAxisGeometry {
        LayoutUnit margin() const { return marginStart.value_or(0_lu) + marginEnd.value_or(0_lu); }

        bool hasNonAutoMargins() const { return marginStart && marginEnd; }

        std::optional<LayoutUnit> definiteSize;

        LayoutUnit ascent;
        LayoutUnit descent;

        std::optional<LayoutUnit> maximumSize;
        std::optional<LayoutUnit> minimumSize;

        std::optional<LayoutUnit> marginStart;
        std::optional<LayoutUnit> marginEnd;

        LayoutUnit borderAndPadding;

        bool hasSizeAuto { false };
    };

    LogicalFlexItem(const ElementBox&, const MainAxisGeometry&, const CrossAxisGeometry&, bool hasAspectRatio, bool isOrhogonal);
    LogicalFlexItem() = default;

    const MainAxisGeometry& mainAxis() const { return m_mainAxisGeometry; }
    const CrossAxisGeometry& crossAxis() const { return m_crossAxisGeometry; }

    float growFactor() const { return style().flexGrow(); }
    float shrinkFactor() const { return style().flexShrink(); }

    bool hasContentFlexBasis() const { return style().flexBasis().isContent(); }
    bool hasAvailableSpaceDependentFlexBasis() const { return false; }
    bool hasAspectRatio() const { return m_hasAspectRatio; }
    bool isOrhogonal() const { return m_isOrhogonal; }
    bool isContentBoxBased() const { return style().boxSizing() == BoxSizing::ContentBox; }

    const ElementBox& layoutBox() const { return *m_layoutBox; }
    const RenderStyle& style() const { return layoutBox().style(); }
    WritingMode writingMode() const { return style().writingMode(); }

private:
    CheckedPtr<const ElementBox> m_layoutBox;

    MainAxisGeometry m_mainAxisGeometry;
    CrossAxisGeometry m_crossAxisGeometry;
    bool m_hasAspectRatio { false };
    bool m_isOrhogonal { false };
};

inline LogicalFlexItem::LogicalFlexItem(const ElementBox& flexItem, const MainAxisGeometry& mainGeometry, const CrossAxisGeometry& crossGeometry, bool hasAspectRatio, bool isOrhogonal)
    : m_layoutBox(flexItem)
    , m_mainAxisGeometry(mainGeometry)
    , m_crossAxisGeometry(crossGeometry)
    , m_hasAspectRatio(hasAspectRatio)
    , m_isOrhogonal(isOrhogonal)
{
}

}
}
