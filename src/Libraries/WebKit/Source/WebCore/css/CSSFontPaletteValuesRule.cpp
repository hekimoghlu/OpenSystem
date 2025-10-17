/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "CSSFontPaletteValuesRule.h"

#include "CSSMarkup.h"
#include "ColorSerialization.h"
#include "JSDOMMapLike.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StyleProperties.h"
#include "StyleRule.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

CSSFontPaletteValuesRule::CSSFontPaletteValuesRule(StyleRuleFontPaletteValues& fontPaletteValuesRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_fontPaletteValuesRule(fontPaletteValuesRule)
{
}

CSSFontPaletteValuesRule::~CSSFontPaletteValuesRule() = default;

String CSSFontPaletteValuesRule::name() const
{
    return m_fontPaletteValuesRule->name();
}

String CSSFontPaletteValuesRule::fontFamily() const
{
    auto serialize = [] (auto& family) {
        return serializeFontFamily(family.string());
    };
    return makeStringByJoining(m_fontPaletteValuesRule->fontFamilies().map(serialize).span(), ", "_s);
}

String CSSFontPaletteValuesRule::basePalette() const
{
    if (!m_fontPaletteValuesRule->basePalette())
        return StringImpl::empty();

    switch (m_fontPaletteValuesRule->basePalette()->type) {
    case FontPaletteIndex::Type::Light:
        return "light"_s;
    case FontPaletteIndex::Type::Dark:
        return "dark"_s;
    case FontPaletteIndex::Type::Integer:
        return makeString(m_fontPaletteValuesRule->basePalette()->integer);
    }
    RELEASE_ASSERT_NOT_REACHED();
}

String CSSFontPaletteValuesRule::overrideColors() const
{     
    StringBuilder result;
    for (size_t i = 0; i < m_fontPaletteValuesRule->overrideColors().size(); ++i) {
        if (i)
            result.append(", "_s);
        const auto& item = m_fontPaletteValuesRule->overrideColors()[i];
        result.append(item.first, ' ', serializationForCSS(item.second));
    }
    return result.toString();
}

String CSSFontPaletteValuesRule::cssText() const
{
    StringBuilder builder;
    builder.append("@font-palette-values "_s, name(), " { "_s);
    if (!m_fontPaletteValuesRule->fontFamilies().isEmpty())
        builder.append("font-family: "_s, fontFamily(), "; "_s);

    if (m_fontPaletteValuesRule->basePalette()) {
        switch (m_fontPaletteValuesRule->basePalette()->type) {
        case FontPaletteIndex::Type::Light:
            builder.append("base-palette: light; "_s);
            break;
        case FontPaletteIndex::Type::Dark:
            builder.append("base-palette: dark; "_s);
            break;
        case FontPaletteIndex::Type::Integer:
            builder.append("base-palette: "_s, m_fontPaletteValuesRule->basePalette()->integer, "; "_s);
            break;
        }
    }

    if (!m_fontPaletteValuesRule->overrideColors().isEmpty()) {
        builder.append("override-colors:"_s);
        for (size_t i = 0; i < m_fontPaletteValuesRule->overrideColors().size(); ++i) {
            if (i)
                builder.append(',');
            builder.append(' ', m_fontPaletteValuesRule->overrideColors()[i].first, ' ', serializationForCSS(m_fontPaletteValuesRule->overrideColors()[i].second));
        }
        builder.append("; "_s);
    }
    builder.append('}');
    return builder.toString();
}

void CSSFontPaletteValuesRule::reattach(StyleRuleBase& rule)
{
    m_fontPaletteValuesRule = downcast<StyleRuleFontPaletteValues>(rule);
}

} // namespace WebCore
