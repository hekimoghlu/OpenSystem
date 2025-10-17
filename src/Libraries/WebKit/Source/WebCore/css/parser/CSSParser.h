/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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

#include "CSSParserContext.h"
#include "CSSParserEnum.h"
#include "CSSProperty.h"
#include "CSSSelectorParser.h"
#include "CSSValue.h"
#include "ColorTypes.h"
#include "WritingMode.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSParserObserver;
class CSSSelectorList;
class Color;
class Element;
class ImmutableStyleProperties;
class MutableStyleProperties;
class StyleRuleBase;
class StyleRuleNestedDeclarations;
class StyleRuleKeyframe;
class StyleSheetContents;

class CSSParser {
public:
    enum class ParseResult {
        Changed,
        Unchanged,
        Error
    };

    WEBCORE_EXPORT explicit CSSParser(const CSSParserContext&);
    WEBCORE_EXPORT ~CSSParser();

    void parseSheet(StyleSheetContents&, const String&);
    
    static RefPtr<StyleRuleBase> parseRule(const CSSParserContext&, StyleSheetContents*, const String&, CSSParserEnum::NestedContext = { });
    
    RefPtr<StyleRuleKeyframe> parseKeyframeRule(const String&);
    static Vector<double> parseKeyframeKeyList(const String&);

    bool parseSupportsCondition(const String&);

    static void parseSheetForInspector(const CSSParserContext&, StyleSheetContents&, const String&, CSSParserObserver&);
    static void parseDeclarationForInspector(const CSSParserContext&, const String&, CSSParserObserver&);

    static ParseResult parseValue(MutableStyleProperties&, CSSPropertyID, const String&, IsImportant, const CSSParserContext&);
    static ParseResult parseCustomPropertyValue(MutableStyleProperties&, const AtomString& propertyName, const String&, IsImportant, const CSSParserContext&);
    
    static RefPtr<CSSValue> parseSingleValue(CSSPropertyID, const String&, const CSSParserContext& = strictCSSParserContext());

    WEBCORE_EXPORT bool parseDeclaration(MutableStyleProperties&, const String&);
    static Ref<ImmutableStyleProperties> parseInlineStyleDeclaration(const String&, const Element&);

    WEBCORE_EXPORT std::optional<CSSSelectorList> parseSelectorList(const String&, StyleSheetContents* = nullptr, CSSParserEnum::NestedContext = { });

    // FIXME: All callers are not getting the right Settings, keyword resolution and calc resolution when using this
    // function and should switch to the parseColorRaw() function in CSSPropertyParserConsumer+Color.h.
    WEBCORE_EXPORT static Color parseColorWithoutContext(const String&, bool strict = false);

    static std::optional<SRGBA<uint8_t>> parseNamedColor(StringView);
    static std::optional<SRGBA<uint8_t>> parseHexColor(StringView);

private:
    ParseResult parseValue(MutableStyleProperties&, CSSPropertyID, const String&, IsImportant);

    CSSParserContext m_context;
};

} // namespace WebCore
