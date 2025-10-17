/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#include "StyleVisitedLinkColorData.h"

#include "RenderStyleInlines.h"
#include "RenderStyleDifference.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleVisitedLinkColorData);

StyleVisitedLinkColorData::StyleVisitedLinkColorData()
    : background(RenderStyle::initialBackgroundColor())
    // borderLeft
    // borderRight
    // borderTop
    // borderBottom
    // textDecoration
    // outline
{
}

StyleVisitedLinkColorData::StyleVisitedLinkColorData(const StyleVisitedLinkColorData& o)
    : RefCounted<StyleVisitedLinkColorData>()
    , background(o.background)
    , borderLeft(o.borderLeft)
    , borderRight(o.borderRight)
    , borderTop(o.borderTop)
    , borderBottom(o.borderBottom)
    , textDecoration(o.textDecoration)
    , outline(o.outline)
{
}

StyleVisitedLinkColorData::~StyleVisitedLinkColorData() = default;

Ref<StyleVisitedLinkColorData> StyleVisitedLinkColorData::copy() const
{
    return adoptRef(*new StyleVisitedLinkColorData(*this));
}

bool StyleVisitedLinkColorData::operator==(const StyleVisitedLinkColorData& o) const
{
    return background == o.background
        && borderLeft == o.borderLeft
        && borderRight == o.borderRight
        && borderTop == o.borderTop
        && borderBottom == o.borderBottom
        && textDecoration == o.textDecoration
        && outline == o.outline;
}

#if !LOG_DISABLED
void StyleVisitedLinkColorData::dumpDifferences(TextStream& ts, const StyleVisitedLinkColorData& other) const
{
    LOG_IF_DIFFERENT(background);
    LOG_IF_DIFFERENT(borderLeft);
    LOG_IF_DIFFERENT(borderRight);
    LOG_IF_DIFFERENT(borderTop);
    LOG_IF_DIFFERENT(borderBottom);
    LOG_IF_DIFFERENT(textDecoration);
    LOG_IF_DIFFERENT(outline);
}
#endif

} // namespace WebCore
