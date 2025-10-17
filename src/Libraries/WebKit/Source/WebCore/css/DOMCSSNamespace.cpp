/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "DOMCSSNamespace.h"

#include "CSSMarkup.h"
#include "CSSParser.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "Document.h"
#include "HighlightRegistry.h"
#include "MutableStyleProperties.h"
#include "StyleProperties.h"
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

bool DOMCSSNamespace::supports(Document& document, const String& property, const String& value)
{
    CSSParserContext parserContext(document);
    parserContext.mode = HTMLStandardMode;

    auto propertyNameWithoutWhitespace = property;
    CSSPropertyID propertyID = cssPropertyID(propertyNameWithoutWhitespace);
    if (propertyID == CSSPropertyInvalid && isCustomPropertyName(propertyNameWithoutWhitespace)) {
        auto dummyStyle = MutableStyleProperties::create();
        return CSSParser::parseCustomPropertyValue(dummyStyle, AtomString { propertyNameWithoutWhitespace }, value, IsImportant::No, parserContext) != CSSParser::ParseResult::Error;
    }

    if (!isExposed(propertyID, &document.settings()))
        return false;

    if (CSSProperty::isDescriptorOnly(propertyID))
        return false;

    if (propertyID == CSSPropertyInvalid)
        return false;

    if (value.isEmpty())
        return false;

    auto dummyStyle = MutableStyleProperties::create();
    return CSSParser::parseValue(dummyStyle, propertyID, value, IsImportant::No, parserContext) != CSSParser::ParseResult::Error;
}

bool DOMCSSNamespace::supports(Document& document, const String& conditionText)
{
    CSSParserContext context(document);
    context.mode = HTMLStandardMode;
    CSSParser parser(context);
    return parser.parseSupportsCondition(conditionText);
}

String DOMCSSNamespace::escape(const String& ident)
{
    StringBuilder builder;
    serializeIdentifier(ident, builder);
    return builder.toString();
}

HighlightRegistry& DOMCSSNamespace::highlights(Document& document)
{
    return document.highlightRegistry();
}

}
