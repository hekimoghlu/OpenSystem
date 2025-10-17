/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include "FloatAvoider.h"

#include "LayoutBox.h"
#include "LayoutElementBox.h"
#include "RenderObject.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FloatAvoider);

// Floating boxes intersect their margin box with the other floats in the context,
// while other float avoiders (e.g. non-floating formatting context roots) intersect their border box.
FloatAvoider::FloatAvoider(LayoutPoint absoluteTopLeft, LayoutUnit borderBoxWidth, const BoxGeometry::Edges& margin, BoxGeometry::HorizontalEdges containingBlockAbsoluteContentBox, bool isFloatingPositioned, bool isStartAligned)
    : m_absoluteTopLeft(absoluteTopLeft)
    , m_borderBoxWidth(borderBoxWidth)
    , m_margin(margin)
    , m_containingBlockAbsoluteContentBox(containingBlockAbsoluteContentBox)
    , m_isFloatingPositioned(isFloatingPositioned)
    , m_isStartAligned(isStartAligned)
{
    m_absoluteTopLeft.setX(initialInlineStart());
}

void FloatAvoider::setInlineStart(LayoutUnit inlineStart)
{
    if (isStartAligned() && isFloatingBox())
        inlineStart += marginStart();
    if (!isStartAligned()) {
        inlineStart -= borderBoxWidth();
        if (isFloatingBox())
            inlineStart -= marginEnd();
    }

    auto constrainedByContainingBlock = [&] {
        // Horizontal position is constrained by the containing block's content box.
        // Compute the horizontal position for the new floating by taking both the contining block and the current left/right floats into account.
        if (isStartAligned())
            return std::max(m_containingBlockAbsoluteContentBox.start + marginStart(), inlineStart);
        // Make sure it does not overflow the containing block on the right.
        return std::min(inlineStart, m_containingBlockAbsoluteContentBox.end - marginBoxWidth() + marginStart());
    }();
    m_absoluteTopLeft.setX(constrainedByContainingBlock);
}

void FloatAvoider::setBlockStart(LayoutUnit blockStart)
{
    if (isFloatingBox())
        blockStart += marginBefore();
    m_absoluteTopLeft.setY(blockStart);
}

LayoutUnit FloatAvoider::initialInlineStart() const
{
    if (isStartAligned())
        return { m_containingBlockAbsoluteContentBox.start + marginStart() };
    return { m_containingBlockAbsoluteContentBox.end - marginEnd() - borderBoxWidth() };
}

bool FloatAvoider::overflowsContainingBlock() const
{
    auto left = m_absoluteTopLeft.x() - marginStart();
    if (m_containingBlockAbsoluteContentBox.start > left)
        return true;

    auto right = left + marginBoxWidth();
    return m_containingBlockAbsoluteContentBox.end < right;
}

}
}
