/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#include "SelectionGeometry.h"

#include "FloatQuad.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SelectionGeometry);

SelectionGeometry::SelectionGeometry(const FloatQuad& quad, SelectionRenderingBehavior behavior, bool isHorizontal, int pageNumber)
    : m_quad(quad)
    , m_behavior(behavior)
    , m_isHorizontal(isHorizontal)
    , m_pageNumber(pageNumber)
{
}

// FIXME: We should move some of these arguments to an auxillary struct.
SelectionGeometry::SelectionGeometry(const FloatQuad& quad, SelectionRenderingBehavior behavior, TextDirection direction, int minX, int maxX, int maxY, int lineNumber, bool isLineBreak, bool isFirstOnLine, bool isLastOnLine, bool containsStart, bool containsEnd, bool isHorizontal, bool isInFixedPosition, int pageNumber)
    : m_quad(quad)
    , m_behavior(behavior)
    , m_direction(direction)
    , m_minX(minX)
    , m_maxX(maxX)
    , m_maxY(maxY)
    , m_lineNumber(lineNumber)
    , m_isLineBreak(isLineBreak)
    , m_isFirstOnLine(isFirstOnLine)
    , m_isLastOnLine(isLastOnLine)
    , m_containsStart(containsStart)
    , m_containsEnd(containsEnd)
    , m_isHorizontal(isHorizontal)
    , m_isInFixedPosition(isInFixedPosition)
    , m_pageNumber(pageNumber)
{
}

SelectionGeometry::SelectionGeometry(const FloatQuad& quad, SelectionRenderingBehavior behavior, TextDirection direction, int minX, int maxX, int maxY, int lineNumber, bool isLineBreak, bool isFirstOnLine, bool isLastOnLine, bool containsStart, bool containsEnd, bool isHorizontal)
    : m_quad(quad)
    , m_behavior(behavior)
    , m_direction(direction)
    , m_minX(minX)
    , m_maxX(maxX)
    , m_maxY(maxY)
    , m_lineNumber(lineNumber)
    , m_isLineBreak(isLineBreak)
    , m_isFirstOnLine(isFirstOnLine)
    , m_isLastOnLine(isLastOnLine)
    , m_containsStart(containsStart)
    , m_containsEnd(containsEnd)
    , m_isHorizontal(isHorizontal)
{
}

void SelectionGeometry::setLogicalLeft(int left)
{
    auto rect = this->rect();
    if (m_isHorizontal)
        rect.setX(left);
    else
        rect.setY(left);
    setRect(rect);
}

void SelectionGeometry::setLogicalWidth(int width)
{
    auto rect = this->rect();
    if (m_isHorizontal)
        rect.setWidth(width);
    else
        rect.setHeight(width);
    setRect(rect);
}

void SelectionGeometry::setLogicalTop(int top)
{
    auto rect = this->rect();
    if (m_isHorizontal)
        rect.setY(top);
    else
        rect.setX(top);
    setRect(rect);
}

void SelectionGeometry::setLogicalHeight(int height)
{
    auto rect = this->rect();
    if (m_isHorizontal)
        rect.setHeight(height);
    else
        rect.setWidth(height);
    setRect(rect);
}

IntRect SelectionGeometry::rect() const
{
    if (!m_cachedEnclosingRect)
        m_cachedEnclosingRect = m_quad.enclosingBoundingBox();
    return *m_cachedEnclosingRect;
}

void SelectionGeometry::setQuad(const FloatQuad& quad)
{
    m_quad = quad;
    m_cachedEnclosingRect.reset();
}

void SelectionGeometry::setRect(const IntRect& rect)
{
    m_quad = FloatQuad { rect };
    m_cachedEnclosingRect = rect;
}

void SelectionGeometry::move(float x, float y)
{
    m_quad.move(x, y);
    m_minX += x;
    m_maxX += x;
    m_maxY += y;
    m_cachedEnclosingRect.reset();
}

TextStream& operator<<(TextStream& stream, const SelectionGeometry& rect)
{
    TextStream::GroupScope group(stream);
    stream << "selection geometry";

    stream.dumpProperty("quad", rect.quad());
    stream.dumpProperty("direction", (rect.direction() == TextDirection::LTR) ? "ltr" : "rtl");

    stream.dumpProperty("min-x", rect.minX());
    stream.dumpProperty("max-x", rect.maxX());
    stream.dumpProperty("max-y", rect.maxY());

    stream.dumpProperty("line number", rect.lineNumber());
    if (rect.isLineBreak())
        stream.dumpProperty("is line break", true);
    if (rect.isFirstOnLine())
        stream.dumpProperty("is first on line", true);
    if (rect.isLastOnLine())
        stream.dumpProperty("is last on line", true);

    if (rect.containsStart())
        stream.dumpProperty("contains start", true);

    if (rect.containsEnd())
        stream.dumpProperty("contains end", true);

    if (rect.isHorizontal())
        stream.dumpProperty("is horizontal", true);

    if (rect.isInFixedPosition())
        stream.dumpProperty("is in fixed position", true);

    if (rect.behavior() == SelectionRenderingBehavior::UseIndividualQuads)
        stream.dumpProperty("using individual quads", true);

    stream.dumpProperty("page number", rect.pageNumber());
    return stream;
}

} // namespace WebCore
