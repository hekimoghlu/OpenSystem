/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "LayoutBoxGeometry.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BoxGeometry);

BoxGeometry::BoxGeometry(const BoxGeometry& other)
    : m_topLeft(other.m_topLeft)
    , m_contentBoxWidth(other.m_contentBoxWidth)
    , m_contentBoxHeight(other.m_contentBoxHeight)
    , m_margin(other.m_margin)
    , m_border(other.m_border)
    , m_padding(other.m_padding)
    , m_verticalSpaceForScrollbar(other.m_verticalSpaceForScrollbar)
    , m_horizontalSpaceForScrollbar(other.m_horizontalSpaceForScrollbar)
#if ASSERT_ENABLED
    , m_hasValidTop(other.m_hasValidTop)
    , m_hasValidLeft(other.m_hasValidLeft)
    , m_hasValidHorizontalMargin(other.m_hasValidHorizontalMargin)
    , m_hasValidVerticalMargin(other.m_hasValidVerticalMargin)
    , m_hasValidBorder(other.m_hasValidBorder)
    , m_hasValidPadding(other.m_hasValidPadding)
    , m_hasValidContentBoxHeight(other.m_hasValidContentBoxHeight)
    , m_hasValidContentBoxWidth(other.m_hasValidContentBoxWidth)
    , m_hasPrecomputedMarginBefore(other.m_hasPrecomputedMarginBefore)
#endif
{
}

BoxGeometry::~BoxGeometry()
{
}

Rect BoxGeometry::marginBox() const
{
    auto borderBox = this->borderBox();

    Rect marginBox;
    marginBox.setTop(borderBox.top() - marginBefore());
    marginBox.setLeft(borderBox.left() - marginStart());
    marginBox.setHeight(borderBox.height() + marginBefore() + marginAfter());
    marginBox.setWidth(borderBox.width() + marginStart() + marginEnd());
    return marginBox;
}

Rect BoxGeometry::borderBox() const
{
    Rect borderBox;
    borderBox.setTopLeft({ });
    borderBox.setSize({ borderBoxWidth(), borderBoxHeight() });
    return borderBox;
}

Rect BoxGeometry::paddingBox() const
{
    auto borderBox = this->borderBox();

    Rect paddingBox;
    paddingBox.setTop(borderBox.top() + borderBefore());
    paddingBox.setLeft(borderBox.left() + borderStart());
    paddingBox.setHeight(borderBox.bottom() - verticalSpaceForScrollbar() - borderAfter() - borderBefore());
    paddingBox.setWidth(borderBox.width() - borderEnd() - horizontalSpaceForScrollbar() - borderStart());
    return paddingBox;
}

Rect BoxGeometry::contentBox() const
{
    Rect contentBox;
    contentBox.setTop(contentBoxTop());
    contentBox.setLeft(contentBoxLeft());
    contentBox.setWidth(contentBoxWidth());
    contentBox.setHeight(contentBoxHeight());
    return contentBox;
}

void BoxGeometry::reset()
{
    setTopLeft({ });

    setHorizontalMargin({ });
    setVerticalMargin({ });
    setBorder({ });
    setPadding({ });

    setContentBoxSize({ });

    setVerticalSpaceForScrollbar({ });
    setHorizontalSpaceForScrollbar({ });
}

}
}

