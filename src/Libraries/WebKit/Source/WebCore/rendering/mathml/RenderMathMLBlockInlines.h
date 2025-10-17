/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

#if ENABLE(MATHML)

#include "RenderBoxInlines.h"
#include "RenderMathMLBlock.h"
#include "RenderTableInlines.h"
#include "StyleInheritedData.h"

namespace WebCore {

inline RenderMathMLTable::RenderMathMLTable(MathMLElement& element, RenderStyle&& style)
    : RenderTable(Type::MathMLTable, element, WTFMove(style))
    , m_mathMLStyle(MathMLStyle::create())
{
    ASSERT(isRenderMathMLTable());
}

inline LayoutUnit RenderMathMLBlock::ascentForChild(const RenderBox& child)
{
    return child.firstLineBaseline().value_or(child.logicalHeight().toInt());
}

inline LayoutUnit RenderMathMLBlock::mirrorIfNeeded(LayoutUnit horizontalOffset, const RenderBox& child) const
{
    return mirrorIfNeeded(horizontalOffset, child.logicalWidth());
}

inline LayoutUnit RenderMathMLBlock::ruleThicknessFallback() const
{
    // This function returns a value for the default rule thickness (TeX's \xi_8) to be used as a fallback when we lack a MATH table.
    // This arbitrary value of 0.05em was used in early WebKit MathML implementations for the thickness of the fraction bars.
    // Note that Gecko has a slower but more accurate version that measures the thickness of U+00AF MACRON to be more accurate and otherwise fallback to some arbitrary value.
    return LayoutUnit(0.05f * style().fontCascade().size());
}

} // namespace WebCore

#endif // ENABLE(MATHML)
