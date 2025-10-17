/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "StylePropertyMap.h"

#include "CSSProperty.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "CSSQuadValue.h"
#include "CSSStyleValueFactory.h"
#include "CSSUnparsedValue.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"
#include "CSSVariableReferenceValue.h"
#include "Document.h"
#include "StylePropertyShorthand.h"
#include <wtf/FixedVector.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static RefPtr<CSSValue> cssValueFromStyleValues(CSSPropertyID propertyID, Vector<Ref<CSSStyleValue>>&& values)
{
    if (values.isEmpty())
        return nullptr;

    auto toCSSValue = [propertyID](CSSStyleValue& styleValue) {
        return styleValue.toCSSValueWithProperty(propertyID);
    };

    if (values.size() == 1)
        return toCSSValue(values[0]);
    CSSValueListBuilder list;
    for (auto&& value : WTFMove(values)) {
        if (auto cssValue = toCSSValue(value))
            list.append(cssValue.releaseNonNull());
    }
    auto separator = CSSProperty::listValuedPropertySeparator(propertyID);
    return CSSValueList::create(separator, WTFMove(list));
}

// https://drafts.css-houdini.org/css-typed-om/#dom-stylepropertymap-set
ExceptionOr<void> StylePropertyMap::set(Document& document, const AtomString& property, FixedVector<std::variant<RefPtr<CSSStyleValue>, String>>&& values)
{
    if (isCustomPropertyName(property)) {
        auto styleValuesOrException = CSSStyleValueFactory::vectorFromStyleValuesOrStrings(property, WTFMove(values), { document });
        if (styleValuesOrException.hasException())
            return styleValuesOrException.releaseException();
        auto styleValues = styleValuesOrException.releaseReturnValue();
        if (styleValues.size() != 1 || !is<CSSUnparsedValue>(styleValues[0].get()))
            return Exception { ExceptionCode::TypeError, "Invalid values"_s };

        auto value = styleValues[0]->toCSSValue();
        if (!value)
            return Exception { ExceptionCode::TypeError, "Invalid values"_s };
        setCustomProperty(document, property, downcast<CSSVariableReferenceValue>(value.releaseNonNull()));
        return { };
    }
    auto propertyID = cssPropertyID(property);
    if (propertyID == CSSPropertyInvalid || !isExposed(propertyID, document.settings()))
        return Exception { ExceptionCode::TypeError, makeString("Invalid property "_s, property) };

    if (!CSSProperty::isListValuedProperty(propertyID) && values.size() > 1)
        return Exception { ExceptionCode::TypeError, makeString(property, " is not a list-valued property but more than one value was provided"_s) };

    if (isShorthand(propertyID)) {
        if (values.size() != 1)
            return Exception { ExceptionCode::TypeError, "Wrong number of values for shorthand CSS property"_s };
        String value;
        switchOn(values[0], [&](const RefPtr<CSSStyleValue>& styleValue) {
            value = styleValue->toString();
        }, [&](const String& string) {
            value = string;
        });
        if (value.isEmpty() || !setShorthandProperty(propertyID, value))
            return Exception { ExceptionCode::TypeError, "Bad value for shorthand CSS property"_s };
        return { };
    }

    auto styleValuesOrException = CSSStyleValueFactory::vectorFromStyleValuesOrStrings(property, WTFMove(values), { document });
    if (styleValuesOrException.hasException())
        return styleValuesOrException.releaseException();
    auto styleValues = styleValuesOrException.releaseReturnValue();
    if (styleValues.size() > 1) {
        for (auto& styleValue : styleValues) {
            if (is<CSSUnparsedValue>(styleValue.get()))
                return Exception { ExceptionCode::TypeError, "There is more than one value and one is either a CSSVariableReferenceValue or a CSSUnparsedValue"_s };
        }
    }
    auto value = cssValueFromStyleValues(propertyID, WTFMove(styleValues));
    if (!value)
        return Exception { ExceptionCode::TypeError, "Invalid values"_s };

    // The CSS Parser may silently convert number values to lengths. However, CSS Typed OM doesn't allow this so
    // we do some pre-validation.
    // FIXME: Eventually, we should be able to generate most of the validation code and not rely on the CSS parser
    // at all.
    if (auto* primitiveValue = dynamicDowncast<CSSPrimitiveValue>(*value); primitiveValue && primitiveValue->isNumberOrInteger()) {
        if (!CSSProperty::allowsNumberOrIntegerInput(propertyID))
            return Exception { ExceptionCode::TypeError, "Invalid value: This property doesn't allow <number> input"_s };
    }

    // FIXME: CSSValuePair has specific behavior related to coalescing its 2 values when they are equal.
    // Throw an error when using them with Typed OM to avoid subtle bugs when the serialization isn't representative of the value.
    if (auto pair = dynamicDowncast<CSSValuePair>(value)) {
        if (pair->canBeCoalesced())
            return Exception { ExceptionCode::NotSupportedError, "Invalid values"_s };
    }

    if (auto quad = dynamicDowncast<CSSQuadValue>(value)) {
        if (quad->canBeCoalesced())
            return Exception { ExceptionCode::TypeError, "Invalid values"_s };
    }

    if (!setProperty(propertyID, value.releaseNonNull()))
        return Exception { ExceptionCode::TypeError, "Invalid values"_s };

    return { };
}

// https://drafts.css-houdini.org/css-typed-om/#dom-stylepropertymap-append
ExceptionOr<void> StylePropertyMap::append(Document& document, const AtomString& property, FixedVector<std::variant<RefPtr<CSSStyleValue>, String>>&& values)
{
    if (values.isEmpty())
        return { };

    if (isCustomPropertyName(property))
        return Exception { ExceptionCode::TypeError, "Cannot append to custom properties"_s };

    auto propertyID = cssPropertyID(property);
    if (propertyID == CSSPropertyInvalid || !isExposed(propertyID, document.settings()))
        return Exception { ExceptionCode::TypeError, makeString("Invalid property "_s, property) };

    if (!CSSProperty::isListValuedProperty(propertyID))
        return Exception { ExceptionCode::TypeError, makeString(property, " does not support multiple values"_s) };

    auto currentValue = propertyValue(propertyID);
    CSSValueListBuilder list;
    if (auto* currentList = dynamicDowncast<CSSValueList>(currentValue.get()))
        list = currentList->copyValues();
    else if (currentValue)
        list.append(currentValue.releaseNonNull());

    auto styleValuesOrException = CSSStyleValueFactory::vectorFromStyleValuesOrStrings(property, WTFMove(values), { document });
    if (styleValuesOrException.hasException())
        return styleValuesOrException.releaseException();

    auto styleValues = styleValuesOrException.releaseReturnValue();
    for (auto& styleValue : styleValues) {
        if (is<CSSUnparsedValue>(styleValue.get()))
            return Exception { ExceptionCode::TypeError, "Values cannot contain a CSSVariableReferenceValue or a CSSUnparsedValue"_s };
        if (auto cssValue = styleValue->toCSSValueWithProperty(propertyID))
            list.append(cssValue.releaseNonNull());
    }

    if (!setProperty(propertyID, CSSValueList::create(CSSProperty::listValuedPropertySeparator(propertyID), WTFMove(list))))
        return Exception { ExceptionCode::TypeError, "Invalid values"_s };

    return { };
}

// https://drafts.css-houdini.org/css-typed-om/#dom-stylepropertymap-delete
ExceptionOr<void> StylePropertyMap::remove(Document& document, const AtomString& property)
{
    if (isCustomPropertyName(property)) {
        removeCustomProperty(property);
        return { };
    }

    auto propertyID = cssPropertyID(property);
    if (!isExposed(propertyID, document.settings()))
        return Exception { ExceptionCode::TypeError, makeString("Invalid property "_s, property) };

    removeProperty(propertyID);
    return { };
}

} // namespace WebCore
