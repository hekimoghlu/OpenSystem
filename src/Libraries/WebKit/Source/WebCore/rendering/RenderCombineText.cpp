/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#include "RenderCombineText.h"

#include "RenderBlock.h"
#include "RenderStyleInlines.h"
#include "StyleInheritedData.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderCombineText);

const float textCombineMargin = 1.15f; // Allow em + 15% margin

RenderCombineText::RenderCombineText(Text& textNode, const String& string)
    : RenderText(Type::CombineText, textNode, string)
    , m_isCombined(false)
    , m_needsFontUpdate(false)
{
    ASSERT(isRenderCombineText());
}

RenderCombineText::~RenderCombineText() = default;

void RenderCombineText::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    // FIXME: This is pretty hackish.
    // Only cache a new font style if our old one actually changed. We do this to avoid
    // clobbering width variants and shrink-to-fit changes, since we won't recombine when
    // the font doesn't change.
    if (!oldStyle || !oldStyle->fontCascadeEqual(style()))
        m_combineFontStyle = RenderStyle::clonePtr(style());

    RenderText::styleDidChange(diff, oldStyle);

    if (m_isCombined && selfNeedsLayout()) {
        // Layouts cause the text to be recombined; therefore, only only un-combine when the style diff causes a layout.
        RenderText::setRenderedText(originalText()); // This RenderCombineText has been combined once. Restore the original text for the next combineText().
        m_isCombined = false;
    }

    m_needsFontUpdate = true;
    combineTextIfNeeded();
}

void RenderCombineText::setRenderedText(const String& text)
{
    RenderText::setRenderedText(text);

    m_needsFontUpdate = true;
    combineTextIfNeeded();
}

std::optional<FloatPoint> RenderCombineText::computeTextOrigin(const FloatRect& boxRect) const
{
    if (!m_isCombined)
        return std::nullopt;

    // Visually center m_combinedTextWidth/Ascent/Descent within boxRect
    FloatPoint result = boxRect.minXMaxYCorner();
    FloatSize combinedTextSize(m_combinedTextWidth, m_combinedTextAscent + m_combinedTextDescent);
    result.move((boxRect.size().transposedSize() - combinedTextSize) / 2);
    result.move(0, m_combinedTextAscent);
    return result;
}

String RenderCombineText::combinedStringForRendering() const
{
    if (m_isCombined) {
        auto originalText = this->originalText();
        ASSERT(!originalText.isNull());
        return originalText;
    }
 
    return { };
}

void RenderCombineText::combineTextIfNeeded()
{
    if (!m_needsFontUpdate)
        return;

    // An ancestor element may trigger us to lay out again, even when we're already combined.
    if (m_isCombined)
        RenderText::setRenderedText(originalText());

    m_isCombined = false;
    m_needsFontUpdate = false;

    // text-combine-upright works only in vertical typographic mode.
    if (!writingMode().isVerticalTypographic())
        return;

    auto description = originalFont().fontDescription();
    float emWidth = description.computedSize() * textCombineMargin;

    RefPtr fontSelector = style().fontCascade().fontSelector();

    description.setOrientation(FontOrientation::Horizontal); // We are going to draw combined text horizontally.

    FontCascade horizontalFont(FontCascadeDescription { description }, style().fontCascade());
    horizontalFont.update(fontSelector.copyRef());

    GlyphOverflow glyphOverflow;
    glyphOverflow.computeBounds = true;
    float combinedTextWidth = width(0, text().length(), horizontalFont, 0, nullptr, &glyphOverflow);

    float bestFitDelta = combinedTextWidth - emWidth;
    auto bestFitDescription = description;

    m_isCombined = combinedTextWidth <= emWidth;
    
    if (m_isCombined)
        m_combineFontStyle->setFontDescription(WTFMove(description)); // Need to change font orientation to horizontal.
    else {
        // Need to try compressed glyphs.
        static const FontWidthVariant widthVariants[] = { FontWidthVariant::HalfWidth, FontWidthVariant::ThirdWidth, FontWidthVariant::QuarterWidth };
        for (auto widthVariant : widthVariants) {
            description.setWidthVariant(widthVariant); // When modifying this, make sure to keep it in sync with FontPlatformData::isForTextCombine()!

            FontCascade compressedFont(FontCascadeDescription { description }, style().fontCascade());
            compressedFont.update(fontSelector.copyRef());
            
            glyphOverflow.left = glyphOverflow.top = glyphOverflow.right = glyphOverflow.bottom = 0;
            float runWidth = width(0, text().length(), compressedFont, 0, nullptr, &glyphOverflow);
            if (runWidth <= emWidth) {
                combinedTextWidth = runWidth;
                m_isCombined = true;

                // Replace my font with the new one.
                m_combineFontStyle->setFontDescription(WTFMove(description));
                break;
            }
            
            float widthDelta = runWidth - emWidth;
            if (widthDelta < bestFitDelta) {
                bestFitDelta = widthDelta;
                bestFitDescription = description;
            }
        }
    }

    if (!m_isCombined) {
        float scaleFactor = std::max(0.4f, emWidth / (emWidth + bestFitDelta));
        float originalSize = bestFitDescription.computedSize();
        do {
            float computedSize = originalSize * scaleFactor;
            bestFitDescription.setComputedSize(computedSize);
            m_combineFontStyle->setFontDescription(FontCascadeDescription { bestFitDescription });
        
            FontCascade compressedFont(FontCascadeDescription { bestFitDescription }, style().fontCascade());
            compressedFont.update(fontSelector.copyRef());
            
            glyphOverflow.left = glyphOverflow.top = glyphOverflow.right = glyphOverflow.bottom = 0;
            float runWidth = width(0, text().length(), compressedFont, 0, nullptr, &glyphOverflow);
            if (runWidth <= emWidth) {
                combinedTextWidth = runWidth;
                m_isCombined = true;
                break;
            }
            scaleFactor -= 0.05f;
        } while (scaleFactor >= 0.4f);
    }

    if (m_isCombined) {
        static NeverDestroyed<String> objectReplacementCharacterString = span(objectReplacementCharacter);
        RenderText::setRenderedText(objectReplacementCharacterString.get());
        m_combinedTextWidth = combinedTextWidth;
        m_combinedTextAscent = glyphOverflow.top;
        m_combinedTextDescent = glyphOverflow.bottom;
        setNeedsLayout();
    }
}

} // namespace WebCore
