/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#include "CSSParser.h"

#include "CSSColorValue.h"
#include "CSSKeyframeRule.h"
#include "CSSParserFastPaths.h"
#include "CSSParserImpl.h"
#include "CSSParserTokenRange.h"
#include "CSSPendingSubstitutionValue.h"
#include "CSSPropertyParser.h"
#include "CSSSelectorParser.h"
#include "CSSSupportsParser.h"
#include "CSSTokenizer.h"
#include "CSSValuePool.h"
#include "Document.h"
#include "Element.h"
#include "ImmutableStyleProperties.h"
#include "MutableStyleProperties.h"
#include "Page.h"
#include "RenderTheme.h"
#include "Settings.h"
#include "StyleColor.h"
#include "StyleRule.h"
#include "StyleSheetContents.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSParser::CSSParser(const CSSParserContext& context)
    : m_context(context)
{
}

CSSParser::~CSSParser() = default;

void CSSParser::parseSheet(StyleSheetContents& sheet, const String& string)
{
    return CSSParserImpl::parseStyleSheet(string, m_context, sheet);
}

void CSSParser::parseSheetForInspector(const CSSParserContext& context, StyleSheetContents& sheet, const String& string, CSSParserObserver& observer)
{
    return CSSParserImpl::parseStyleSheetForInspector(string, context, sheet, observer);
}

RefPtr<StyleRuleBase> CSSParser::parseRule(const CSSParserContext& context, StyleSheetContents* sheet, const String& string, CSSParserEnum::NestedContext nestedContext)
{
    return CSSParserImpl::parseRule(string, context, sheet, CSSParserImpl::AllowedRules::ImportRules, nestedContext);
}

RefPtr<StyleRuleKeyframe> CSSParser::parseKeyframeRule(const String& string)
{
    RefPtr keyframe = CSSParserImpl::parseRule(string, m_context, nullptr, CSSParserImpl::AllowedRules::KeyframeRules);
    return downcast<StyleRuleKeyframe>(keyframe.get());
}

bool CSSParser::parseSupportsCondition(const String& condition)
{
    CSSParserImpl parser(m_context, condition);
    if (!parser.tokenizer())
        return false;
    return CSSSupportsParser::supportsCondition(parser.tokenizer()->tokenRange(), parser, CSSSupportsParser::ParsingMode::AllowBareDeclarationAndGeneralEnclosed) == CSSSupportsParser::Supported;
}

static Color color(RefPtr<CSSValue>&& value)
{
    if (!value)
        return { };
    return CSSColorValue::absoluteColor(*value);
}

Color CSSParser::parseColorWithoutContext(const String& string, bool strict)
{
    if (auto color = CSSParserFastPaths::parseSimpleColor(string, strict))
        return *color;
    // FIXME: Unclear why we want to ignore the boolean argument "strict" and always pass strictCSSParserContext here.
    return color(parseSingleValue(CSSPropertyColor, string, strictCSSParserContext()));
}

std::optional<SRGBA<uint8_t>> CSSParser::parseNamedColor(StringView string)
{
    return CSSParserFastPaths::parseNamedColor(string);
}

std::optional<SRGBA<uint8_t>> CSSParser::parseHexColor(StringView string)
{
    return CSSParserFastPaths::parseHexColor(string);
}

RefPtr<CSSValue> CSSParser::parseSingleValue(CSSPropertyID propertyID, const String& string, const CSSParserContext& context)
{
    if (string.isEmpty())
        return nullptr;
    if (RefPtr value = CSSParserFastPaths::maybeParseValue(propertyID, string, context))
        return value;
    CSSTokenizer tokenizer(string);
    return CSSPropertyParser::parseSingleValue(propertyID, tokenizer.tokenRange(), context);
}

CSSParser::ParseResult CSSParser::parseValue(MutableStyleProperties& declaration, CSSPropertyID propertyID, const String& string, IsImportant important, const CSSParserContext& context)
{
    ASSERT(!string.isEmpty());
    if (RefPtr value = CSSParserFastPaths::maybeParseValue(propertyID, string, context))
        return declaration.addParsedProperty(CSSProperty(propertyID, WTFMove(value), important)) ? CSSParser::ParseResult::Changed : CSSParser::ParseResult::Unchanged;
    CSSParser parser(context);
    return parser.parseValue(declaration, propertyID, string, important);
}

CSSParser::ParseResult CSSParser::parseCustomPropertyValue(MutableStyleProperties& declaration, const AtomString& propertyName, const String& string, IsImportant important, const CSSParserContext& context)
{
    return CSSParserImpl::parseCustomPropertyValue(declaration, propertyName, string, important, context);
}

CSSParser::ParseResult CSSParser::parseValue(MutableStyleProperties& declaration, CSSPropertyID propertyID, const String& string, IsImportant important)
{
    return CSSParserImpl::parseValue(declaration, propertyID, string, important, m_context);
}

std::optional<CSSSelectorList> CSSParser::parseSelectorList(const String& string, StyleSheetContents* styleSheet, CSSParserEnum::NestedContext nestedContext)
{
    return parseCSSSelectorList(CSSTokenizer(string).tokenRange(), m_context, styleSheet, nestedContext);
}

Ref<ImmutableStyleProperties> CSSParser::parseInlineStyleDeclaration(const String& string, const Element& element)
{
    return CSSParserImpl::parseInlineStyleDeclaration(string, element);
}

bool CSSParser::parseDeclaration(MutableStyleProperties& declaration, const String& string)
{
    return CSSParserImpl::parseDeclarationList(&declaration, string, m_context);
}

void CSSParser::parseDeclarationForInspector(const CSSParserContext& context, const String& string, CSSParserObserver& observer)
{
    CSSParserImpl::parseDeclarationListForInspector(string, context, observer);
}

Vector<double> CSSParser::parseKeyframeKeyList(const String& selector)
{
    return CSSParserImpl::parseKeyframeKeyList(selector);
}

}
