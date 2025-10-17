/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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

#include "InlineIteratorSVGTextBox.h"
#include "LegacyRenderSVGResource.h"
#include "TextBoxPainter.h"

namespace WebCore {

class RenderBoxModelObject;
class RenderSVGInlineText;
class RenderStyle;
class SVGPaintServerHandling;
struct SVGTextFragment;

template<typename TextBoxPath>
class SVGTextBoxPainter {
public:
    SVGTextBoxPainter(TextBoxPath&&, PaintInfo&, const LayoutPoint& paintOffset);
    ~SVGTextBoxPainter() = default;

    void paint();
    void paintSelectionBackground();

private:
    InlineIterator::SVGTextBoxIterator textBoxIterator() const;

    const RenderSVGInlineText& renderer() const { return m_renderer; }
    const RenderBoxModelObject& parentRenderer() const;
    OptionSet<RenderSVGResourceMode> paintingResourceMode() const { return m_paintingResourceMode; }

    void paintDecoration(OptionSet<TextDecorationLine>, const SVGTextFragment&);
    void paintDecorationWithStyle(OptionSet<TextDecorationLine>, const SVGTextFragment&, const RenderBoxModelObject&);
    void paintTextWithShadows(const RenderStyle&, TextRun&, const SVGTextFragment&, unsigned startPosition, unsigned endPosition);
    void paintText(const RenderStyle&, const RenderStyle& selectionStyle, const SVGTextFragment&, bool hasSelection, bool paintSelectedTextOnly);

    bool acquirePaintingResource(SVGPaintServerHandling&, float scalingFactor, const RenderBoxModelObject&, const RenderStyle&);
    void releasePaintingResource(SVGPaintServerHandling&);

    bool acquireLegacyPaintingResource(GraphicsContext*&, float scalingFactor, RenderBoxModelObject&, const RenderStyle&);
    void releaseLegacyPaintingResource(GraphicsContext*&, const Path*);

    std::pair<unsigned, unsigned> selectionStartEnd() const;

    const TextBoxPath m_textBox;
    const RenderSVGInlineText& m_renderer;
    PaintInfo& m_paintInfo;
    const TextBoxSelectableRange m_selectableRange;
    const LayoutPoint m_paintOffset;
    const bool m_haveSelection;

    OptionSet<RenderSVGResourceMode> m_paintingResourceMode;
    LegacyRenderSVGResource* m_legacyPaintingResource { nullptr };
};

class LegacySVGTextBoxPainter : public SVGTextBoxPainter<InlineIterator::BoxLegacyPath> {
public:
    LegacySVGTextBoxPainter(const SVGInlineTextBox&, PaintInfo&, const LayoutPoint& paintOffset);
};

class ModernSVGTextBoxPainter : public SVGTextBoxPainter<InlineIterator::BoxModernPath> {
public:
    ModernSVGTextBoxPainter(const LayoutIntegration::InlineContent&, size_t boxIndex, PaintInfo&, const LayoutPoint& paintOffset);
};

TextRun constructTextRun(StringView, TextDirection, const RenderStyle&, const SVGTextFragment&);
FloatRect selectionRectForTextFragment(const RenderSVGInlineText&, TextDirection, const SVGTextFragment&, unsigned startPosition, unsigned endPosition, const RenderStyle&);
bool mapStartEndPositionsIntoFragmentCoordinates(unsigned textBoxStart, const SVGTextFragment&, unsigned& startPosition, unsigned& endPosition);

}
