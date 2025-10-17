/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

#include "FloatRoundedRect.h"
#include "LayoutShape.h"
#include "RenderStyleConstants.h"

namespace WebCore {

class RenderBox;

RoundedRect computeRoundedRectForBoxShape(CSSBoxType, const RenderBox&);

class BoxLayoutShape final : public LayoutShape {
public:
    BoxLayoutShape(const FloatRoundedRect& bounds)
        : m_bounds(bounds)
    {
    }

    LayoutRect shapeMarginLogicalBoundingBox() const override;
    bool isEmpty() const override { return m_bounds.isEmpty(); }
    LineSegment getExcludedInterval(LayoutUnit logicalTop, LayoutUnit logicalHeight) const override;

    void buildDisplayPaths(DisplayPaths&) const override;

private:
    FloatRoundedRect shapeMarginBounds() const;

    FloatRoundedRect m_bounds;
};

} // namespace WebCore
