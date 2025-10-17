/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "StylePropertyMapReadOnly.h"

#include "CSSCustomPropertyValue.h"
#include "CSSImageValue.h"
#include "CSSPrimitiveValue.h"
#include "CSSProperty.h"
#include "CSSPropertyNames.h"
#include "CSSStyleImageValue.h"
#include "CSSStyleValueFactory.h"
#include "CSSUnitValue.h"
#include "CSSUnparsedValue.h"
#include "CSSValueList.h"
#include "Document.h"

namespace WebCore {

RefPtr<CSSStyleValue> StylePropertyMapReadOnly::reifyValue(RefPtr<CSSValue>&& value, std::optional<CSSPropertyID> propertyID, Document& document)
{
    if (!value)
        return nullptr;
    auto result = CSSStyleValueFactory::reifyValue(value.releaseNonNull(), propertyID, &document);
    return (result.hasException() ? nullptr : RefPtr<CSSStyleValue> { result.releaseReturnValue() });
}

Vector<RefPtr<CSSStyleValue>> StylePropertyMapReadOnly::reifyValueToVector(RefPtr<CSSValue>&& value, std::optional<CSSPropertyID> propertyID, Document& document)
{
    if (!value)
        return { };

    if (auto* customPropertyValue = dynamicDowncast<CSSCustomPropertyValue>(*value)) {
        if (std::holds_alternative<CSSCustomPropertyValue::SyntaxValueList>(customPropertyValue->value())) {
            auto& list = std::get<CSSCustomPropertyValue::SyntaxValueList>(customPropertyValue->value());

            Vector<RefPtr<CSSStyleValue>> result;
            result.reserveInitialCapacity(list.values.size());
            for (auto& listValue : list.values) {
                auto styleValue = CSSStyleValueFactory::constructStyleValueForCustomPropertySyntaxValue(listValue);
                if (!styleValue)
                    return { };
                result.append(WTFMove(styleValue));
            }
            return result;
        }
    }

    auto* valueList = dynamicDowncast<CSSValueList>(*value);
    if (!valueList || (propertyID && !CSSProperty::isListValuedProperty(*propertyID)))
        return { StylePropertyMapReadOnly::reifyValue(WTFMove(value), propertyID, document) };

    return WTF::map(*valueList, [&](auto& item) {
        return StylePropertyMapReadOnly::reifyValue(Ref { const_cast<CSSValue&>(item) }, propertyID, document);
    });
}

StylePropertyMapReadOnly::Iterator::Iterator(StylePropertyMapReadOnly& map, ScriptExecutionContext* context)
    : m_values(map.entries(context))
{
}

std::optional<StylePropertyMapReadOnly::StylePropertyMapEntry> StylePropertyMapReadOnly::Iterator::next()
{
    if (m_index >= m_values.size())
        return std::nullopt;

    return m_values[m_index++];
}

} // namespace WebCore
