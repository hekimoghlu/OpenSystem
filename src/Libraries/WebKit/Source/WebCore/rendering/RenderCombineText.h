/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
#include "Text.h"

namespace WebCore {

class RenderCombineText final : public RenderText {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderCombineText);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderCombineText);
public:
    RenderCombineText(Text&, const String&);
    virtual ~RenderCombineText();

    Text& textNode() const { return downcast<Text>(nodeForNonAnonymous()); }

    void combineTextIfNeeded();
    std::optional<FloatPoint> computeTextOrigin(const FloatRect& boxRect) const;
    String combinedStringForRendering() const;
    bool isCombined() const { return m_isCombined; }
    float combinedTextWidth(const FontCascade& font) const { return font.size(); }
    const FontCascade& originalFont() const { return parent()->style().fontCascade(); }
    const FontCascade& textCombineFont() const { return m_combineFontStyle->fontCascade(); }

private:
    void node() const = delete;

    ASCIILiteral renderName() const override { return "RenderCombineText"_s; }
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;
    void setRenderedText(const String&) override;

    std::unique_ptr<RenderStyle> m_combineFontStyle;
    float m_combinedTextWidth { 0 };
    float m_combinedTextAscent { 0 };
    float m_combinedTextDescent { 0 };
    bool m_isCombined : 1;
    bool m_needsFontUpdate : 1;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderCombineText, isRenderCombineText())
