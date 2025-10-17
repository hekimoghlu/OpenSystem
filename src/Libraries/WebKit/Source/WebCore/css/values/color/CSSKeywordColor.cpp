/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#include "CSSKeywordColor.h"

#include "CSSColorType.h"
#include "CSSPlatformColorResolutionState.h"
#include "CSSValueKeywords.h"
#include "HashTools.h"
#include "RenderTheme.h"
#include "StyleColorOptions.h"
#include <wtf/OptionSet.h>

namespace WebCore {
namespace CSS {

static bool isVGAPaletteColor(CSSValueID id)
{
    // https://drafts.csswg.org/css-color-4/#named-colors
    // "16 of CSSâ€™s named colors come from the VGA palette originally, and were then adopted into HTML"
    return id >= CSSValueAqua && id <= CSSValueGrey;
}

static bool isNonVGANamedColor(CSSValueID id)
{
    // https://drafts.csswg.org/css-color-4/#named-colors
    return id >= CSSValueAliceblue && id <= CSSValueYellowgreen;
}

bool isAbsoluteColorKeyword(CSSValueID id)
{
    // https://drafts.csswg.org/css-color-4/#typedef-absolute-color
    return isVGAPaletteColor(id) || isNonVGANamedColor(id) || id == CSSValueAlpha || id == CSSValueTransparent;
}

bool isCurrentColorKeyword(CSSValueID id)
{
    return id == CSSValueCurrentcolor;
}

bool isSystemColorKeyword(CSSValueID id)
{
    // https://drafts.csswg.org/css-color-4/#css-system-colors
    return (id >= CSSValueCanvas && id <= CSSValueInternalDocumentTextColor) || id == CSSValueText || isDeprecatedSystemColorKeyword(id);
}

bool isDeprecatedSystemColorKeyword(CSSValueID id)
{
    // https://drafts.csswg.org/css-color-4/#deprecated-system-colors
    return (id >= CSSValueActiveborder && id <= CSSValueWindowtext) || id == CSSValueMenu;
}

bool isColorKeyword(CSSValueID id, OptionSet<ColorType> allowedColorTypes)
{
    return (allowedColorTypes.contains(ColorType::Absolute) && isAbsoluteColorKeyword(id))
        || (allowedColorTypes.contains(ColorType::Current) && isCurrentColorKeyword(id))
        || (allowedColorTypes.contains(ColorType::System) && isSystemColorKeyword(id));
}

bool isColorKeyword(CSSValueID id)
{
    return isAbsoluteColorKeyword(id) || isCurrentColorKeyword(id) || isSystemColorKeyword(id);
}

WebCore::Color colorFromAbsoluteKeyword(CSSValueID keyword)
{
    ASSERT(isAbsoluteColorKeyword(keyword));

    // TODO: Investigate if this should be a constexpr map for performance.

    if (auto valueName = nameLiteral(keyword)) {
        if (auto namedColor = findColor(valueName.characters(), valueName.length()))
            return asSRGBA(PackedColor::ARGB { namedColor->ARGBValue });
    }
    ASSERT_NOT_REACHED();
    return { };
}

WebCore::Color colorFromKeyword(CSSValueID keyword, OptionSet<StyleColorOptions> options)
{
    if (isAbsoluteColorKeyword(keyword))
        return colorFromAbsoluteKeyword(keyword);

    return RenderTheme::singleton().systemColor(keyword, options);
}

WebCore::Color createColor(const KeywordColor& unresolved, PlatformColorResolutionState& state)
{
    switch (unresolved.valueID) {
    case CSSValueInternalDocumentTextColor:
        return state.internalDocumentTextColor();
    case CSSValueWebkitLink:
        return state.forVisitedLink == Style::ForVisitedLink::Yes ? state.webkitLinkVisited() : state.webkitLink();
    case CSSValueWebkitActivelink:
        return state.webkitActiveLink();
    case CSSValueWebkitFocusRingColor:
        return state.webkitFocusRingColor();
    case CSSValueCurrentcolor:
        return state.currentColor();
    default:
        return colorFromKeyword(unresolved.valueID, state.keywordOptions);
    }
}

bool containsCurrentColor(const KeywordColor& unresolved)
{
    return isCurrentColorKeyword(unresolved.valueID);
}

bool containsColorSchemeDependentColor(const KeywordColor& unresolved)
{
    return isSystemColorKeyword(unresolved.valueID);
}

void Serialize<KeywordColor>::operator()(StringBuilder& builder, const KeywordColor& value)
{
    builder.append(nameLiteralForSerialization(value.valueID));
}

} // namespace CSS
} // namespace WebCore
