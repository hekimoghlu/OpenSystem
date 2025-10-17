/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

#include "RenderBlock.h"
#include "ScrollTypes.h"

namespace WebCore {

class RenderScrollbar;

class RenderScrollbarPart final : public RenderBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderScrollbarPart);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderScrollbarPart);
public:
    RenderScrollbarPart(Document&, RenderStyle&&, RenderScrollbar* = nullptr, ScrollbarPart = NoPart);
    
    virtual ~RenderScrollbarPart();

    ASCIILiteral renderName() const override { return "RenderScrollbarPart"_s; }
    
    bool requiresLayer() const override { return false; }

    void layout() override;
    
    void paintIntoRect(GraphicsContext&, const LayoutPoint&, const LayoutRect&);

    // Scrollbar parts needs to be rendered at device pixel boundaries.
    LayoutUnit marginTop() const override { ASSERT(isIntegerValue(m_marginBox.top())); return m_marginBox.top(); }
    LayoutUnit marginBottom() const override { ASSERT(isIntegerValue(m_marginBox.bottom())); return m_marginBox.bottom(); }
    LayoutUnit marginLeft() const override { ASSERT(isIntegerValue(m_marginBox.left())); return m_marginBox.left(); }
    LayoutUnit marginRight() const override { ASSERT(isIntegerValue(m_marginBox.right())); return m_marginBox.right(); }

    RenderBox* rendererOwningScrollbar() const;

private:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;
    void imageChanged(WrappedImagePtr, const IntRect* = nullptr) override;

    void layoutHorizontalPart();
    void layoutVerticalPart();

    void computeScrollbarWidth();
    void computeScrollbarHeight();
    
    SingleThreadWeakPtr<RenderScrollbar> m_scrollbar;
    ScrollbarPart m_part;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderScrollbarPart, isRenderScrollbarPart())
