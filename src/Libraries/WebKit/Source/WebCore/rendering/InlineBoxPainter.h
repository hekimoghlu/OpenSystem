/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

#include "FloatRect.h"
#include "GraphicsTypes.h"
#include "InlineIteratorInlineBox.h"
#include "LayoutRect.h"
#include "ShadowData.h"

namespace WebCore {

class Color;
class FillLayer;
class LegacyInlineFlowBox;
class RenderBoxModelObject;
class RenderStyle;
struct PaintInfo;

class InlineBoxPainter {
public:
    InlineBoxPainter(const LegacyInlineFlowBox&, PaintInfo&, const LayoutPoint& paintOffset);
    InlineBoxPainter(const LayoutIntegration::InlineContent&, const InlineDisplay::Box&, PaintInfo&, const LayoutPoint& paintOffset);
    ~InlineBoxPainter();

    void paint();

private:
    InlineBoxPainter(const InlineIterator::InlineBox&, PaintInfo&, const LayoutPoint& paintOffset);

    void paintMask();
    void paintDecorations();
    void paintFillLayers(const Color&, const FillLayer&, const LayoutRect& paintRect, CompositeOperator);
    void paintFillLayer(const Color&, const FillLayer&, const LayoutRect& paintRect, CompositeOperator);
    void paintBoxShadow(ShadowStyle, const LayoutRect& paintRect);

    const RenderStyle& style() const;
    // FIXME: Make RenderBoxModelObject functions const.
    RenderBoxModelObject& renderer() const { return const_cast<RenderBoxModelObject&>(m_renderer); }
    bool isHorizontal() const { return m_isHorizontal; }

    const InlineIterator::InlineBox m_inlineBox;
    PaintInfo& m_paintInfo;
    const LayoutPoint m_paintOffset;
    const RenderBoxModelObject& m_renderer;
    const bool m_isFirstLineBox;
    const bool m_isRootInlineBox;
    const bool m_isHorizontal;
};

}
