/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#include "FormattingGeometry.h"

namespace WebCore {
namespace Layout {

class BlockFormattingContext;

// This class implements positioning and sizing for boxes participating in a block formatting context.
class BlockFormattingGeometry : public FormattingGeometry {
public:
    BlockFormattingGeometry(const BlockFormattingContext&);

    ContentHeightAndMargin inFlowContentHeightAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin inFlowContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    LayoutUnit staticVerticalPosition(const ElementBox&, LayoutUnit containingBlockContentBoxTop) const;
    LayoutUnit staticHorizontalPosition(const ElementBox&, const HorizontalConstraints&) const;

    IntrinsicWidthConstraints intrinsicWidthConstraints(const ElementBox&) const;

    ContentWidthAndMargin computedContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, std::optional<LayoutUnit> availableWidthFloatAvoider) const;

private:
    ContentHeightAndMargin inFlowNonReplacedContentHeightAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenVerticalValues&) const;
    ContentWidthAndMargin inFlowNonReplacedContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;
    ContentWidthAndMargin inFlowReplacedContentWidthAndMargin(const ElementBox&, const HorizontalConstraints&, const OverriddenHorizontalValues&) const;

    const BlockFormattingContext& formattingContext() const { return downcast<BlockFormattingContext>(FormattingGeometry::formattingContext()); }
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_GEOMETRY(BlockFormattingGeometry, isBlockFormattingGeometry())

