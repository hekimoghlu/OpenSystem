/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include "CSSStyleValue.h"

#include "CSSParser.h"
#include "CSSPropertyParser.h"
#include "CSSStyleValueFactory.h"
#include "CSSUnitValue.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringView.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSStyleValue);

ExceptionOr<Ref<CSSStyleValue>> CSSStyleValue::parse(const Document& document, const AtomString& property, const String& cssText)
{
    constexpr bool parseMultiple = false;
    auto parseResult = CSSStyleValueFactory::parseStyleValue(property, cssText, parseMultiple, { document });
    if (parseResult.hasException())
        return parseResult.releaseException();
    
    auto returnValue = parseResult.releaseReturnValue();
    
    // Returned vector should not be empty. If parsing failed, an exception should be returned.
    if (returnValue.isEmpty())
        return Exception { ExceptionCode::SyntaxError, makeString(cssText, " cannot be parsed as a "_s, property) };

    return WTFMove(returnValue.at(0));
}

ExceptionOr<Vector<Ref<CSSStyleValue>>> CSSStyleValue::parseAll(const Document& document, const AtomString& property, const String& cssText)
{
    constexpr bool parseMultiple = true;
    return CSSStyleValueFactory::parseStyleValue(property, cssText, parseMultiple, { document });
}

Ref<CSSStyleValue> CSSStyleValue::create(RefPtr<CSSValue>&& cssValue, String&& property)
{
    return adoptRef(*new CSSStyleValue(WTFMove(cssValue), WTFMove(property)));
}

Ref<CSSStyleValue> CSSStyleValue::create()
{
    return adoptRef(*new CSSStyleValue());
}

CSSStyleValue::CSSStyleValue(RefPtr<CSSValue>&& cssValue, String&& property)
    : m_customPropertyName(WTFMove(property))
    , m_propertyValue(WTFMove(cssValue))
{
}

String CSSStyleValue::toString() const
{
    StringBuilder builder;
    serialize(builder);
    return builder.toString();
}

void CSSStyleValue::serialize(StringBuilder& builder, OptionSet<SerializationArguments>) const
{
    if (m_propertyValue)
        builder.append(m_propertyValue->cssText());
}

} // namespace WebCore
