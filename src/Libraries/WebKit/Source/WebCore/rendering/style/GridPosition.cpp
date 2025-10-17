/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#include "GridPosition.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

std::optional<int> GridPosition::gMaxPositionForTesting;
static const int kGridMaxPosition = 1000000;

void GridPosition::setExplicitPosition(int position, const String& namedGridLine)
{
    m_type = GridPositionType::ExplicitPosition;
    setIntegerPosition(position);
    m_namedGridLine = namedGridLine;
}

void GridPosition::setAutoPosition()
{
    m_type = GridPositionType::AutoPosition;
    m_integerPosition = 0;
}

// 'span' values cannot be negative, yet we reuse the <integer> position which can
// be. This means that we have to convert the span position to an integer, losing
// some precision here. It shouldn't be an issue in practice though.
void GridPosition::setSpanPosition(int position, const String& namedGridLine)
{
    m_type = GridPositionType::SpanPosition;
    setIntegerPosition(position);
    m_namedGridLine = namedGridLine;
}

void GridPosition::setNamedGridArea(const String& namedGridArea)
{
    m_type = GridPositionType::NamedGridAreaPosition;
    m_namedGridLine = namedGridArea;
}

int GridPosition::integerPosition() const
{
    ASSERT(type() == GridPositionType::ExplicitPosition);
    return m_integerPosition;
}

String GridPosition::namedGridLine() const
{
    ASSERT(type() == GridPositionType::ExplicitPosition || type() == GridPositionType::SpanPosition || type() == GridPositionType::NamedGridAreaPosition);
    return m_namedGridLine;
}

int GridPosition::spanPosition() const
{
    ASSERT(type() == GridPositionType::SpanPosition);
    return m_integerPosition;
}

int GridPosition::max()
{
    return gMaxPositionForTesting.value_or(kGridMaxPosition);
}

int GridPosition::min()
{
    return -max();
}

void GridPosition::setMaxPositionForTesting(unsigned maxPosition)
{
    gMaxPositionForTesting = static_cast<int>(maxPosition);
}

TextStream& operator<<(TextStream& ts, const GridPosition& o)
{
    switch (o.type()) {
    case GridPositionType::AutoPosition:
        return ts << "auto";
    case GridPositionType::ExplicitPosition:
        return ts << o.namedGridLine() << " " << o.integerPosition();
    case GridPositionType::SpanPosition:
        return ts << "span" << " " << o.namedGridLine() << " " << o.integerPosition();
    case GridPositionType::NamedGridAreaPosition:
        return ts << o.namedGridLine();
    }
    return ts;
}

} // namespace WebCore
