/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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

#include "CSSCustomPropertySyntax.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "StyleRuleType.h"
#include <wtf/text/StringView.h>

namespace WebCore {

class CSSCustomPropertyValue;
class CSSProperty;
class StylePropertyShorthand;

namespace Style {
class BuilderState;
}
    
// Inputs: PropertyID, isImportant bool, CSSParserTokenRange.
// Outputs: Vector of CSSProperties

class CSSPropertyParser {
    WTF_MAKE_NONCOPYABLE(CSSPropertyParser);
public:
    static bool parseValue(CSSPropertyID, bool important, const CSSParserTokenRange&, const CSSParserContext&, Vector<CSSProperty, 256>&, StyleRuleType);

    // Parses a non-shorthand CSS property
    static RefPtr<CSSValue> parseSingleValue(CSSPropertyID, const CSSParserTokenRange&, const CSSParserContext&);

    static RefPtr<CSSCustomPropertyValue> parseTypedCustomPropertyInitialValue(const AtomString&, const CSSCustomPropertySyntax&, CSSParserTokenRange, Style::BuilderState&, const CSSParserContext&);
    static RefPtr<CSSCustomPropertyValue> parseTypedCustomPropertyValue(const AtomString& name, const CSSCustomPropertySyntax&, const CSSParserTokenRange&, Style::BuilderState&, const CSSParserContext&);
    static ComputedStyleDependencies collectParsedCustomPropertyValueDependencies(const CSSCustomPropertySyntax&, const CSSParserTokenRange&, const CSSParserContext&);
    static bool isValidCustomPropertyValueForSyntax(const CSSCustomPropertySyntax&, CSSParserTokenRange, const CSSParserContext&);

    static RefPtr<CSSValue> parseCounterStyleDescriptor(CSSPropertyID, CSSParserTokenRange&, const CSSParserContext&);

private:
    CSSPropertyParser(const CSSParserTokenRange&, const CSSParserContext&, Vector<CSSProperty, 256>*, bool consumeWhitespace = true);

    // FIXME: Rename once the CSSParserValue-based parseValue is removed
    bool parseValueStart(CSSPropertyID, bool important);
    bool consumeCSSWideKeyword(CSSPropertyID, bool important);
    RefPtr<CSSValue> parseSingleValue(CSSPropertyID, CSSPropertyID = CSSPropertyInvalid);
    
    std::pair<RefPtr<CSSValue>, CSSCustomPropertySyntax::Type> consumeCustomPropertyValueWithSyntax(const CSSCustomPropertySyntax&);
    RefPtr<CSSCustomPropertyValue> parseTypedCustomPropertyValue(const AtomString& name, const CSSCustomPropertySyntax&, Style::BuilderState&);
    ComputedStyleDependencies collectParsedCustomPropertyValueDependencies(const CSSCustomPropertySyntax&);

    bool inQuirksMode() const { return m_context.mode == HTMLQuirksMode; }

    // @font-face descriptors.
    bool parseFontFaceDescriptor(CSSPropertyID);
    bool parseFontFaceDescriptorShorthand(CSSPropertyID);

    // @font-palette-values descriptors.
    bool parseFontPaletteValuesDescriptor(CSSPropertyID);

    // @counter-style descriptors.
    bool parseCounterStyleDescriptor(CSSPropertyID);
    
    // @keyframe descriptors.
    bool parseKeyframeDescriptor(CSSPropertyID, bool important);

    // @page descriptors.
    bool parsePageDescriptor(CSSPropertyID, bool important);

    // @property descriptors.
    bool parsePropertyDescriptor(CSSPropertyID);

    // @view-transition descriptors.
    bool parseViewTransitionDescriptor(CSSPropertyID);

    // @position-try descriptors.
    bool parsePositionTryDescriptor(CSSPropertyID, bool important);

    void addProperty(CSSPropertyID longhand, CSSPropertyID shorthand, RefPtr<CSSValue>&&, bool important, bool implicit = false);
    void addExpandedProperty(CSSPropertyID shorthand, RefPtr<CSSValue>&&, bool important, bool implicit = false);

    // Shorthand Parsing.

    bool parseShorthand(CSSPropertyID, bool important);
    bool consumeShorthandGreedily(const StylePropertyShorthand&, bool important);
    bool consume2ValueShorthand(const StylePropertyShorthand&, bool important);
    bool consume4ValueShorthand(const StylePropertyShorthand&, bool important);

    bool consumeBorderShorthand(CSSPropertyID widthProperty, CSSPropertyID styleProperty, CSSPropertyID colorProperty, bool important);

    // Legacy parsing allows <string>s for animation-name
    bool consumeAnimationShorthand(const StylePropertyShorthand&, bool important);
    bool consumeBackgroundShorthand(const StylePropertyShorthand&, bool important);
    bool consumeOverflowShorthand(bool important);

    bool consumeColumns(bool important);

    bool consumeGridItemPositionShorthand(CSSPropertyID, bool important);
    bool consumeGridTemplateRowsAndAreasAndColumns(CSSPropertyID, bool important);
    bool consumeGridTemplateShorthand(CSSPropertyID, bool important);
    bool consumeGridShorthand(bool important);
    bool consumeGridAreaShorthand(bool important);

    bool consumeAlignShorthand(const StylePropertyShorthand&, bool important);

    bool consumeBlockStepShorthand(bool important);

    bool consumeFont(bool important);
    bool consumeTextDecorationSkip(bool important);
    bool consumeFontVariantShorthand(bool important);
    bool consumeFontSynthesis(bool important);

    bool consumeBorderSpacing(bool important);

    // CSS3 Parsing Routines (for properties specific to CSS3)
    bool consumeBorderImage(CSSPropertyID, bool important);
    bool consumeBorderRadius(CSSPropertyID, bool important);

    bool consumeFlex(bool important);

    bool consumeLegacyBreakProperty(CSSPropertyID, bool important);
    bool consumeLegacyTextOrientation(bool important);

    bool consumeTransformOrigin(bool important);
    bool consumePerspectiveOrigin(bool important);
    bool consumePrefixedPerspective(bool important);
    bool consumeOffset(bool important);
    bool consumeListStyleShorthand(bool important);

    bool consumeOverscrollBehaviorShorthand(bool important);

    bool consumeContainerShorthand(bool important);
    bool consumeContainIntrinsicSizeShorthand(bool important);

    bool consumeAnimationRangeShorthand(bool important);
    bool consumeScrollTimelineShorthand(bool important);
    bool consumeViewTimelineShorthand(bool important);

    bool consumeLineClampShorthand(bool important);

    bool consumeTextBoxShorthand(bool important);

    bool consumeTextWrapShorthand(bool important);
    bool consumeWhiteSpaceShorthand(bool important);

private:
    // Inputs:
    CSSParserTokenRange m_range;
    const CSSParserContext& m_context;

    // Outputs:
    Vector<CSSProperty, 256>* m_parsedProperties;
};

CSSPropertyID cssPropertyID(StringView);
WEBCORE_EXPORT CSSValueID cssValueKeywordID(StringView);
bool isCustomPropertyName(StringView);

bool isInitialValueForLonghand(CSSPropertyID, const CSSValue&);
ASCIILiteral initialValueTextForLonghand(CSSPropertyID);
CSSValueID initialValueIDForLonghand(CSSPropertyID); // Returns CSSPropertyInvalid if not a keyword.

} // namespace WebCore
