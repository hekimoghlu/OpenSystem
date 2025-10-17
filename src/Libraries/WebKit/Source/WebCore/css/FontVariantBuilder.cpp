/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include "FontVariantBuilder.h"

#include "CSSPrimitiveValue.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"
#include "TextFlags.h"

namespace WebCore {

FontVariantLigaturesValues extractFontVariantLigatures(const CSSValue& value)
{
    FontVariantLigatures common = FontVariantLigatures::Normal;
    FontVariantLigatures discretionary = FontVariantLigatures::Normal;
    FontVariantLigatures historical = FontVariantLigatures::Normal;
    FontVariantLigatures contextualAlternates = FontVariantLigatures::Normal;

    if (auto* valueList = dynamicDowncast<CSSValueList>(value)) {
        for (auto& item : *valueList) {
            switch (item.valueID()) {
            case CSSValueNoCommonLigatures:
                common = FontVariantLigatures::No;
                break;
            case CSSValueCommonLigatures:
                common = FontVariantLigatures::Yes;
                break;
            case CSSValueNoDiscretionaryLigatures:
                discretionary = FontVariantLigatures::No;
                break;
            case CSSValueDiscretionaryLigatures:
                discretionary = FontVariantLigatures::Yes;
                break;
            case CSSValueNoHistoricalLigatures:
                historical = FontVariantLigatures::No;
                break;
            case CSSValueHistoricalLigatures:
                historical = FontVariantLigatures::Yes;
                break;
            case CSSValueContextual:
                contextualAlternates = FontVariantLigatures::Yes;
                break;
            case CSSValueNoContextual:
                contextualAlternates = FontVariantLigatures::No;
                break;
            default:
                ASSERT_NOT_REACHED();
                break;
            }
        }
    } else if (is<CSSPrimitiveValue>(value)) {
        switch (value.valueID()) {
        case CSSValueNormal:
            break;
        case CSSValueNone:
            common = FontVariantLigatures::No;
            discretionary = FontVariantLigatures::No;
            historical = FontVariantLigatures::No;
            contextualAlternates = FontVariantLigatures::No;
            break;
        default:
            ASSERT_NOT_REACHED();
            break;
        }
    }

    return FontVariantLigaturesValues(common, discretionary, historical, contextualAlternates);
}

FontVariantNumericValues extractFontVariantNumeric(const CSSValue& value)
{
    FontVariantNumericFigure figure = FontVariantNumericFigure::Normal;
    FontVariantNumericSpacing spacing = FontVariantNumericSpacing::Normal;
    FontVariantNumericFraction fraction = FontVariantNumericFraction::Normal;
    FontVariantNumericOrdinal ordinal = FontVariantNumericOrdinal::Normal;
    FontVariantNumericSlashedZero slashedZero = FontVariantNumericSlashedZero::Normal;

    if (auto* valueList = dynamicDowncast<CSSValueList>(value)) {
        for (auto& item : *valueList) {
            switch (item.valueID()) {
            case CSSValueLiningNums:
                figure = FontVariantNumericFigure::LiningNumbers;
                break;
            case CSSValueOldstyleNums:
                figure = FontVariantNumericFigure::OldStyleNumbers;
                break;
            case CSSValueProportionalNums:
                spacing = FontVariantNumericSpacing::ProportionalNumbers;
                break;
            case CSSValueTabularNums:
                spacing = FontVariantNumericSpacing::TabularNumbers;
                break;
            case CSSValueDiagonalFractions:
                fraction = FontVariantNumericFraction::DiagonalFractions;
                break;
            case CSSValueStackedFractions:
                fraction = FontVariantNumericFraction::StackedFractions;
                break;
            case CSSValueOrdinal:
                ordinal = FontVariantNumericOrdinal::Yes;
                break;
            case CSSValueSlashedZero:
                slashedZero = FontVariantNumericSlashedZero::Yes;
                break;
            default:
                ASSERT_NOT_REACHED();
                break;
            }
        }
    } else if (is<CSSPrimitiveValue>(value))
        ASSERT(value.valueID() == CSSValueNormal);

    return FontVariantNumericValues(figure, spacing, fraction, ordinal, slashedZero);
}

FontVariantEastAsianValues extractFontVariantEastAsian(const CSSValue& value)
{
    FontVariantEastAsianVariant variant = FontVariantEastAsianVariant::Normal;
    FontVariantEastAsianWidth width = FontVariantEastAsianWidth::Normal;
    FontVariantEastAsianRuby ruby = FontVariantEastAsianRuby::Normal;

    if (auto* valueList = dynamicDowncast<CSSValueList>(value)) {
        for (auto& item : *valueList) {
            switch (item.valueID()) {
            case CSSValueJis78:
                variant = FontVariantEastAsianVariant::Jis78;
                break;
            case CSSValueJis83:
                variant = FontVariantEastAsianVariant::Jis83;
                break;
            case CSSValueJis90:
                variant = FontVariantEastAsianVariant::Jis90;
                break;
            case CSSValueJis04:
                variant = FontVariantEastAsianVariant::Jis04;
                break;
            case CSSValueSimplified:
                variant = FontVariantEastAsianVariant::Simplified;
                break;
            case CSSValueTraditional:
                variant = FontVariantEastAsianVariant::Traditional;
                break;
            case CSSValueFullWidth:
                width = FontVariantEastAsianWidth::Full;
                break;
            case CSSValueProportionalWidth:
                width = FontVariantEastAsianWidth::Proportional;
                break;
            case CSSValueRuby:
                ruby = FontVariantEastAsianRuby::Yes;
                break;
            default:
                ASSERT_NOT_REACHED();
                break;
            }
        }
    } else if (is<CSSPrimitiveValue>(value))
        ASSERT(value.valueID() == CSSValueNormal);

    return FontVariantEastAsianValues(variant, width, ruby);
}

} // WebCore namespace
