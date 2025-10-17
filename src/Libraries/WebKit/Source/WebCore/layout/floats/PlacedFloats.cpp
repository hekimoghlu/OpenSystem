/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "PlacedFloats.h"

#include "LayoutContainingBlockChainIterator.h"
#include "LayoutInitialContainingBlock.h"
#include "LayoutShape.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PlacedFloats);

PlacedFloats::Item::Item(const Box& layoutBox, Position position, const BoxGeometry& absoluteBoxGeometry, LayoutPoint localTopLeft, std::optional<size_t> line)
    : m_layoutBox(layoutBox)
    , m_position(position)
    , m_absoluteBoxGeometry(absoluteBoxGeometry)
    , m_localTopLeft(localTopLeft)
    , m_shape(layoutBox.shape())
    , m_placedByLine(line)
{
}

PlacedFloats::Item::Item(Position position, const BoxGeometry& absoluteBoxGeometry, LayoutPoint localTopLeft, const LayoutShape* shape)
    : m_position(position)
    , m_absoluteBoxGeometry(absoluteBoxGeometry)
    , m_localTopLeft(localTopLeft)
    , m_shape(shape)
{
}

BoxGeometry PlacedFloats::Item::boxGeometry() const
{
    auto boxGeometry = BoxGeometry { m_absoluteBoxGeometry };
    boxGeometry.setTopLeft(m_localTopLeft);
    return boxGeometry;
}

PlacedFloats::Item::~Item() = default;

PlacedFloats::PlacedFloats(const ElementBox& blockFormattingContextRoot)
    : m_blockFormattingContextRoot(blockFormattingContextRoot)
    , m_writingMode(blockFormattingContextRoot.writingMode())
{
    ASSERT(blockFormattingContextRoot.establishesBlockFormattingContext());
}

void PlacedFloats::append(Item newFloatItem)
{
    auto isStartPositioned = newFloatItem.isStartPositioned();
    m_positionTypes.add(isStartPositioned ? PositionType::Start : PositionType::End);

    if (m_list.isEmpty())
        return m_list.append(newFloatItem);

    // The integration codepath does not construct a layout box for the float item.
    ASSERT_IMPLIES(newFloatItem.layoutBox(), m_list.findIf([&] (auto& entry) {
        return entry.layoutBox() == newFloatItem.layoutBox();
    }) == notFound);

    // When adding a new float item to the list, we have to ensure that it is definitely the left(right)-most item.
    // Normally it is, but negative inline-axis margins can push the float box beyond another float box.
    // Float items in m_list list should stay in inline-axis position order (left/right edge) on the same block-axis position.
    auto inlineAxisMargin = newFloatItem.inlineAxisMargin();
    auto hasNegativeInlineAxisMargin = (isStartPositioned && inlineAxisMargin.start < 0) || (!isStartPositioned && inlineAxisMargin.end < 0);
    if (!hasNegativeInlineAxisMargin)
        return m_list.append(newFloatItem);

    for (size_t i = m_list.size(); i--;) {
        auto& floatItem = m_list[i];
        if (isStartPositioned != floatItem.isStartPositioned())
            continue;

        auto isInlineAxisOrdered = [&] {
            if (newFloatItem.absoluteRectWithMargin().top() > floatItem.absoluteRectWithMargin().top()) {
                // There's no more floats on this block axis position.
                return true;
            }
            return (isStartPositioned && newFloatItem.absoluteRectWithMargin().right() >= floatItem.absoluteRectWithMargin().right())
                || (!isStartPositioned && newFloatItem.absoluteRectWithMargin().left() <= floatItem.absoluteRectWithMargin().left());
        };
        if (isInlineAxisOrdered())
            return m_list.insert(i + 1, newFloatItem);
    }
    m_list.insert(0, newFloatItem);
}

bool PlacedFloats::Item::isInFormattingContextOf(const ElementBox& formattingContextRoot) const
{
    ASSERT(formattingContextRoot.establishesFormattingContext());
    ASSERT(!is<InitialContainingBlock>(m_layoutBox));
    for (auto& containingBlock : containingBlockChain(*m_layoutBox)) {
        if (&containingBlock == &formattingContextRoot)
            return true;
    }
    ASSERT_NOT_REACHED();
    return false;
}

void PlacedFloats::clear()
{
    m_list.clear();
    m_positionTypes = { };
}

std::optional<LayoutUnit> PlacedFloats::highestPositionOnBlockAxis() const
{
    auto highestBlockAxisPosition = std::optional<LayoutUnit> { };
    for (auto& floatItem : m_list)
        highestBlockAxisPosition = !highestBlockAxisPosition ? floatItem.absoluteRectWithMargin().top() : std::min(*highestBlockAxisPosition, floatItem.absoluteRectWithMargin().top());
    return highestBlockAxisPosition;
}

std::optional<LayoutUnit> PlacedFloats::lowestPositionOnBlockAxis(Clear type) const
{
    // TODO: Currently this is only called once for each formatting context root with floats per layout.
    // Cache the value if we end up calling it more frequently (and update it at append/remove).
    auto lowestBlockAxisPosition = std::optional<LayoutUnit> { };
    for (auto& floatItem : m_list) {
        if ((type == Clear::InlineStart && !floatItem.isStartPositioned()) || (type == Clear::InlineEnd && floatItem.isStartPositioned()))
            continue;
        lowestBlockAxisPosition = !lowestBlockAxisPosition ? floatItem.absoluteRectWithMargin().bottom() : std::max(*lowestBlockAxisPosition, floatItem.absoluteRectWithMargin().bottom());
    }
    return lowestBlockAxisPosition;
}

void PlacedFloats::shrinkToFit()
{
    m_list.shrinkToFit();
}

}
}
