/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "StyleBoxData.h"

#include "RenderStyleConstants.h"
#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

struct SameSizeAsStyleBoxData : public RefCounted<SameSizeAsStyleBoxData> {
    Length length[7];
    int m_zIndex[2];
    uint32_t bitfields;
};

static_assert(sizeof(StyleBoxData) == sizeof(SameSizeAsStyleBoxData), "StyleBoxData should not grow");

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleBoxData);

StyleBoxData::StyleBoxData()
    : m_minWidth(RenderStyle::initialMinSize())
    , m_maxWidth(RenderStyle::initialMaxSize())
    , m_minHeight(RenderStyle::initialMinSize())
    , m_maxHeight(RenderStyle::initialMaxSize())
    , m_specifiedZIndex(0)
    , m_usedZIndex(0)
    , m_hasAutoSpecifiedZIndex(true)
    , m_hasAutoUsedZIndex(true)
    , m_boxSizing(static_cast<unsigned>(BoxSizing::ContentBox))
    , m_boxDecorationBreak(static_cast<unsigned>(BoxDecorationBreak::Slice))
    , m_verticalAlign(static_cast<unsigned>(RenderStyle::initialVerticalAlign()))
{
}

inline StyleBoxData::StyleBoxData(const StyleBoxData& o)
    : RefCounted<StyleBoxData>()
    , m_width(o.m_width)
    , m_height(o.m_height)
    , m_minWidth(o.m_minWidth)
    , m_maxWidth(o.m_maxWidth)
    , m_minHeight(o.m_minHeight)
    , m_maxHeight(o.m_maxHeight)
    , m_verticalAlignLength(o.m_verticalAlignLength)
    , m_specifiedZIndex(o.m_specifiedZIndex)
    , m_usedZIndex(o.m_usedZIndex)
    , m_hasAutoSpecifiedZIndex(o.m_hasAutoSpecifiedZIndex)
    , m_hasAutoUsedZIndex(o.m_hasAutoUsedZIndex)
    , m_boxSizing(o.m_boxSizing)
    , m_boxDecorationBreak(o.m_boxDecorationBreak)
    , m_verticalAlign(o.m_verticalAlign)
{
}

Ref<StyleBoxData> StyleBoxData::copy() const
{
    return adoptRef(*new StyleBoxData(*this));
}

bool StyleBoxData::operator==(const StyleBoxData& o) const
{
    return m_width == o.m_width
        && m_height == o.m_height
        && m_minWidth == o.m_minWidth
        && m_maxWidth == o.m_maxWidth
        && m_minHeight == o.m_minHeight
        && m_maxHeight == o.m_maxHeight
        && m_verticalAlignLength == o.m_verticalAlignLength
        && m_specifiedZIndex == o.m_specifiedZIndex
        && m_hasAutoSpecifiedZIndex == o.m_hasAutoSpecifiedZIndex
        && m_usedZIndex == o.m_usedZIndex
        && m_hasAutoUsedZIndex == o.m_hasAutoUsedZIndex
        && m_boxSizing == o.m_boxSizing
        && m_boxDecorationBreak == o.m_boxDecorationBreak
        && m_verticalAlign == o.m_verticalAlign;
}

#if !LOG_DISABLED
void StyleBoxData::dumpDifferences(TextStream& ts, const StyleBoxData& other) const
{
    LOG_IF_DIFFERENT(m_width);
    LOG_IF_DIFFERENT(m_height);

    LOG_IF_DIFFERENT(m_minWidth);
    LOG_IF_DIFFERENT(m_maxWidth);

    LOG_IF_DIFFERENT(m_minHeight);
    LOG_IF_DIFFERENT(m_maxHeight);

    LOG_IF_DIFFERENT(m_verticalAlignLength);

    LOG_IF_DIFFERENT(m_specifiedZIndex);
    LOG_IF_DIFFERENT(m_usedZIndex);

    LOG_IF_DIFFERENT_WITH_CAST(bool, m_hasAutoSpecifiedZIndex);
    LOG_IF_DIFFERENT_WITH_CAST(bool, m_hasAutoUsedZIndex);

    LOG_IF_DIFFERENT_WITH_CAST(BoxSizing, m_boxSizing);
    LOG_IF_DIFFERENT_WITH_CAST(BoxDecorationBreak, m_boxDecorationBreak);
    LOG_IF_DIFFERENT_WITH_CAST(VerticalAlign, m_verticalAlign);
}
#endif

} // namespace WebCore
