/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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

#if ENABLE(MATHML)

#include "RenderMathMLBlock.h"

namespace WebCore {

class MathMLTokenElement;

class RenderMathMLToken : public RenderMathMLBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLToken);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLToken);
public:
    RenderMathMLToken(Type, MathMLTokenElement&, RenderStyle&&);
    RenderMathMLToken(Type, Document&, RenderStyle&&);
    virtual ~RenderMathMLToken();

    MathMLTokenElement& element();

    virtual void updateTokenContent();
    void updateFromElement() override;

protected:
    void paint(PaintInfo&, const LayoutPoint&) override;
    void paintChildren(PaintInfo& forSelf, const LayoutPoint&, PaintInfo& forChild, bool usePrintRect) override;
    std::optional<LayoutUnit> firstLineBaseline() const override;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) override;
    void computePreferredLogicalWidths() override;

private:
    bool isRenderMathMLToken() const final { return true; }
    ASCIILiteral renderName() const override { return "RenderMathMLToken"_s; }
    bool isChildAllowed(const RenderObject&, const RenderStyle&) const final { return true; };
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;
    void updateMathVariantGlyph();
    void setMathVariantGlyphDirty()
    {
        m_mathVariantGlyphDirty = true;
        setNeedsLayoutAndPrefWidthsRecalc();
    }
    std::optional<char32_t> m_mathVariantCodePoint { std::nullopt };
    bool m_mathVariantIsMirrored { false };
    bool m_mathVariantGlyphDirty { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLToken, isRenderMathMLToken())

#endif // ENABLE(MATHML)
