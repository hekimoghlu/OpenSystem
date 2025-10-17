/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "StyleBuilderState.h"

#include "CSSAppleColorFilterPropertyValue.h"
#include "CSSCanvasValue.h"
#include "CSSColorValue.h"
#include "CSSCrossfadeValue.h"
#include "CSSCursorImageValue.h"
#include "CSSFilterImageValue.h"
#include "CSSFilterPropertyValue.h"
#include "CSSFontSelector.h"
#include "CSSFunctionValue.h"
#include "CSSGradientValue.h"
#include "CSSImageSetValue.h"
#include "CSSImageValue.h"
#include "CSSNamedImageValue.h"
#include "CSSPaintImageValue.h"
#include "CalculationRandomKeyMap.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "ElementInlines.h"
#include "FontCache.h"
#include "HTMLElement.h"
#include "RenderStyleSetters.h"
#include "RenderTheme.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSVGElement.h"
#include "Settings.h"
#include "StyleAppleColorFilterProperty.h"
#include "StyleBuilder.h"
#include "StyleCachedImage.h"
#include "StyleCanvasImage.h"
#include "StyleColor.h"
#include "StyleCrossfadeImage.h"
#include "StyleCursorImage.h"
#include "StyleFilterImage.h"
#include "StyleFilterProperty.h"
#include "StyleFontSizeFunctions.h"
#include "StyleGeneratedImage.h"
#include "StyleGradientImage.h"
#include "StyleImageSet.h"
#include "StyleNamedImage.h"
#include "StylePaintImage.h"
#include "TransformOperationsBuilder.h"

namespace WebCore {
namespace Style {

BuilderState::BuilderState(Builder& builder, RenderStyle& style, BuilderContext&& context)
    : m_builder(builder)
    , m_styleMap(*this)
    , m_style(style)
    , m_context(WTFMove(context))
    , m_cssToLengthConversionData(style, *this)
{
}

// SVG handles zooming in a different way compared to CSS. The whole document is scaled instead
// of each individual length value in the render style / tree. CSSPrimitiveValue::resolveAsLength*()
// multiplies each resolved length with the zoom multiplier - so for SVG we need to disable that.
// Though all CSS values that can be applied to outermost <svg> elements (width/height/border/padding...)
// need to respect the scaling. RenderBox (the parent class of LegacyRenderSVGRoot) grabs values like
// width/height/border/padding/... from the RenderStyle -> for SVG these values would never scale,
// if we'd pass a 1.0 zoom factor everyhwere. So we only pass a zoom factor of 1.0 for specific
// properties that are NOT allowed to scale within a zoomed SVG document (letter/word-spacing/font-size).
bool BuilderState::useSVGZoomRules() const
{
    return is<SVGElement>(element());
}

bool BuilderState::useSVGZoomRulesForLength() const
{
    return is<SVGElement>(element()) && !(is<SVGSVGElement>(*element()) && element()->parentNode());
}

RefPtr<StyleImage> BuilderState::createStyleImage(const CSSValue& value) const
{
    if (auto* imageValue = dynamicDowncast<CSSImageValue>(value))
        return imageValue->createStyleImage(*this);
    if (auto* imageSetValue = dynamicDowncast<CSSImageSetValue>(value))
        return imageSetValue->createStyleImage(*this);
    if (auto* imageValue = dynamicDowncast<CSSCursorImageValue>(value))
        return imageValue->createStyleImage(*this);
    if (auto* imageValue = dynamicDowncast<CSSNamedImageValue>(value))
        return imageValue->createStyleImage(*this);
    if (auto* cssCanvasValue = dynamicDowncast<CSSCanvasValue>(value))
        return cssCanvasValue->createStyleImage(*this);
    if (auto* crossfadeValue = dynamicDowncast<CSSCrossfadeValue>(value))
        return crossfadeValue->createStyleImage(*this);
    if (auto* filterImageValue = dynamicDowncast<CSSFilterImageValue>(value))
        return filterImageValue->createStyleImage(*this);
    if (auto* gradientValue = dynamicDowncast<CSSGradientValue>(value))
        return gradientValue->createStyleImage(*this);
    if (auto* paintImageValue = dynamicDowncast<CSSPaintImageValue>(value))
        return paintImageValue->createStyleImage(*this);
    return nullptr;
}

FilterOperations BuilderState::createFilterOperations(const CSS::FilterProperty& value) const
{
    return WebCore::Style::createFilterOperations(value, document(), m_style, m_cssToLengthConversionData);
}

FilterOperations BuilderState::createFilterOperations(const CSSValue& value) const
{
    if (RefPtr primitive = dynamicDowncast<CSSPrimitiveValue>(value)) {
        ASSERT(primitive->valueID() == CSSValueNone);
        return { };
    }

    Ref filterValue = downcast<CSSFilterPropertyValue>(value);
    return createFilterOperations(filterValue->filter());
}

FilterOperations BuilderState::createAppleColorFilterOperations(const CSS::AppleColorFilterProperty& value) const
{
    return WebCore::Style::createAppleColorFilterOperations(value, document(), m_style, m_cssToLengthConversionData);
}

FilterOperations BuilderState::createAppleColorFilterOperations(const CSSValue& value) const
{
    if (RefPtr primitive = dynamicDowncast<CSSPrimitiveValue>(value)) {
        ASSERT(primitive->valueID() == CSSValueNone);
        return { };
    }

    Ref filterValue = downcast<CSSAppleColorFilterPropertyValue>(value);
    return createAppleColorFilterOperations(filterValue->filter());
}

Color BuilderState::createStyleColor(const CSSValue& value, ForVisitedLink forVisitedLink) const
{
    if (!element() || !element()->isLink())
        forVisitedLink = ForVisitedLink::No;

    if (RefPtr color = dynamicDowncast<CSSColorValue>(value))
        return toStyle(color->color(), *this, forVisitedLink);
    return toStyle(CSS::Color { CSS::KeywordColor { value.valueID() } }, *this, forVisitedLink);
}

void BuilderState::registerContentAttribute(const AtomString& attributeLocalName)
{
    if (style().pseudoElementType() == PseudoId::Before || style().pseudoElementType() == PseudoId::After)
        m_registeredContentAttributes.append(attributeLocalName);
}

void BuilderState::adjustStyleForInterCharacterRuby()
{
    if (!m_style.isInterCharacterRubyPosition() || !element() || !element()->hasTagName(HTMLNames::rtTag))
        return;

    m_style.setTextAlign(TextAlignMode::Center);
    if (!m_style.writingMode().isVerticalTypographic())
        m_style.setWritingMode(StyleWritingMode::VerticalLr);
}

void BuilderState::updateFont()
{
    auto& fontSelector = const_cast<Document&>(document()).fontSelector();

    auto needsUpdate = [&] {
        if (m_fontDirty)
            return true;
        auto* fonts = m_style.fontCascade().fonts();
        if (!fonts)
            return true;
        return false;
    };

    if (!needsUpdate())
        return;

#if ENABLE(TEXT_AUTOSIZING)
    updateFontForTextSizeAdjust();
#endif
    updateFontForGenericFamilyChange();
    updateFontForZoomChange();
    updateFontForOrientationChange();

    m_style.fontCascade().update(&fontSelector);

    m_fontDirty = false;
}

#if ENABLE(TEXT_AUTOSIZING)
void BuilderState::updateFontForTextSizeAdjust()
{
    if (m_style.textSizeAdjust().isAuto()
        || !document().settings().textAutosizingEnabled()
        || (document().settings().textAutosizingUsesIdempotentMode()
            && !m_style.textSizeAdjust().isNone()
            && !document().settings().idempotentModeAutosizingOnlyHonorsPercentages()))
        return;

    auto newFontDescription = m_style.fontDescription();
    if (!m_style.textSizeAdjust().isNone())
        newFontDescription.setComputedSize(newFontDescription.specifiedSize() * m_style.textSizeAdjust().multiplier());
    else
        newFontDescription.setComputedSize(newFontDescription.specifiedSize());

    m_style.setFontDescriptionWithoutUpdate(WTFMove(newFontDescription));
}
#endif

void BuilderState::updateFontForZoomChange()
{
    if (m_style.usedZoom() == parentStyle().usedZoom() && m_style.textZoom() == parentStyle().textZoom())
        return;

    const auto& childFont = m_style.fontDescription();
    auto newFontDescription = childFont;
    setFontSize(newFontDescription, childFont.specifiedSize());

    m_style.setFontDescriptionWithoutUpdate(WTFMove(newFontDescription));
}

void BuilderState::updateFontForGenericFamilyChange()
{
    const auto& childFont = m_style.fontDescription();

    if (childFont.isAbsoluteSize())
        return;

    const auto& parentFont = parentStyle().fontDescription();
    if (childFont.useFixedDefaultSize() == parentFont.useFixedDefaultSize())
        return;

    // We know the parent is monospace or the child is monospace, and that font
    // size was unspecified. We want to scale our font size as appropriate.
    // If the font uses a keyword size, then we refetch from the table rather than
    // multiplying by our scale factor.
    float size = [&] {
        if (CSSValueID sizeIdentifier = childFont.keywordSizeAsIdentifier())
            return Style::fontSizeForKeyword(sizeIdentifier, childFont.useFixedDefaultSize(), document());

        auto fixedSize =  document().settings().defaultFixedFontSize();
        auto defaultSize =  document().settings().defaultFontSize();
        float fixedScaleFactor = (fixedSize && defaultSize) ? static_cast<float>(fixedSize) / defaultSize : 1;
        return parentFont.useFixedDefaultSize() ? childFont.specifiedSize() / fixedScaleFactor : childFont.specifiedSize() * fixedScaleFactor;
    }();

    auto newFontDescription = childFont;
    setFontSize(newFontDescription, size);
    m_style.setFontDescriptionWithoutUpdate(WTFMove(newFontDescription));
}

void BuilderState::updateFontForOrientationChange()
{
    auto [fontOrientation, glyphOrientation] = m_style.fontAndGlyphOrientation();

    const auto& fontDescription = m_style.fontDescription();
    if (fontDescription.orientation() == fontOrientation && fontDescription.nonCJKGlyphOrientation() == glyphOrientation)
        return;

    auto newFontDescription = fontDescription;
    newFontDescription.setNonCJKGlyphOrientation(glyphOrientation);
    newFontDescription.setOrientation(fontOrientation);
    m_style.setFontDescriptionWithoutUpdate(WTFMove(newFontDescription));
}

void BuilderState::setFontSize(FontCascadeDescription& fontDescription, float size)
{
    fontDescription.setSpecifiedSize(size);
    fontDescription.setComputedSize(Style::computedFontSizeFromSpecifiedSize(size, fontDescription.isAbsoluteSize(), useSVGZoomRules(), &style(), document()));
}

CSSPropertyID BuilderState::cssPropertyID() const
{
    return m_currentProperty ? m_currentProperty->id : CSSPropertyInvalid;
}

bool BuilderState::isCurrentPropertyInvalidAtComputedValueTime() const
{
    return m_invalidAtComputedValueTimeProperties.get(cssPropertyID());
}

void BuilderState::setCurrentPropertyInvalidAtComputedValueTime()
{
    m_invalidAtComputedValueTimeProperties.set(cssPropertyID());
}

Ref<Calculation::RandomKeyMap> BuilderState::randomKeyMap(bool perElement) const
{
    if (perElement) {
        ASSERT(element());

        std::optional<Style::PseudoElementIdentifier> pseudoElementIdentifier;
        if (style().pseudoElementType() != PseudoId::None)
            pseudoElementIdentifier = Style::PseudoElementIdentifier { style().pseudoElementType(), style().pseudoElementNameArgument() };

        return element()->randomKeyMap(pseudoElementIdentifier);
    }
    return document().randomKeyMap();
}

} // namespace Style
} // namespace WebCore
