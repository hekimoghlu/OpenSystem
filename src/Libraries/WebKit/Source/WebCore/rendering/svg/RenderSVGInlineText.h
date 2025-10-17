/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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

#include "FontCascade.h"
#include "RenderText.h"
#include "SVGTextLayoutAttributes.h"
#include "Text.h"

namespace WebCore {

class SVGInlineTextBox;

class RenderSVGInlineText final : public RenderText {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGInlineText);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGInlineText);
public:
    RenderSVGInlineText(Text&, const String&);
    virtual ~RenderSVGInlineText();

    Text& textNode() const { return downcast<Text>(nodeForNonAnonymous()); }

    bool characterStartsNewTextChunk(int position) const;
    SVGTextLayoutAttributes* layoutAttributes() { return &m_layoutAttributes; }
    const SVGTextLayoutAttributes* layoutAttributes() const { return &m_layoutAttributes; }

    // computeScalingFactor() returns the font-size scaling factor, ignoring the text-rendering mode.
    // scalingFactor() takes it into account, and thus returns 1 whenever text-rendering is set to 'geometricPrecision'.
    // Therefore if you need access to the vanilla scaling factor, use this method directly (e.g. for non-scaling-stroke).
    static float computeScalingFactorForRenderer(const RenderObject&);

    float scalingFactor() const { return m_scalingFactor; }
    const FontCascade& scaledFont() const { return m_scaledFont; }
    void updateScaledFont();
    static bool computeNewScaledFontForStyle(const RenderObject&, const RenderStyle&, float& scalingFactor, FontCascade& scaledFont);

    // Preserves floating point precision for the use in DRT. It knows how to round and does a better job than enclosingIntRect.
    FloatRect floatLinesBoundingBox() const;

private:
    ASCIILiteral renderName() const override { return "RenderSVGInlineText"_s; }

    String originalText() const override;
    void setRenderedText(const String&) override;
    void styleDidChange(StyleDifference, const RenderStyle*) override;

    FloatRect objectBoundingBox() const override { return floatLinesBoundingBox(); }

    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) override;
    IntRect linesBoundingBox() const override;
    std::unique_ptr<LegacyInlineTextBox> createTextBox() override;

    float m_scalingFactor;
    FontCascade m_scaledFont;
    SVGTextLayoutAttributes m_layoutAttributes;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGInlineText, isRenderSVGInlineText())
