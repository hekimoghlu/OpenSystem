/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
#include "DeprecatedCSSOMValue.h"

#include "DeprecatedCSSOMBoxShadowValue.h"
#include "DeprecatedCSSOMFilterFunctionValue.h"
#include "DeprecatedCSSOMPrimitiveValue.h"
#include "DeprecatedCSSOMTextShadowValue.h"
#include "DeprecatedCSSOMValueList.h"

namespace WebCore {

void DeprecatedCSSOMValue::operator delete(DeprecatedCSSOMValue* value, std::destroying_delete_t)
{
    auto destroyAndFree = [&]<typename ValueType> (ValueType& value) {
        std::destroy_at(&value);
        ValueType::freeAfterDestruction(&value);
    };

    switch (value->classType()) {
    case ClassType::BoxShadow:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMBoxShadowValue>(*value));
        break;
    case ClassType::Complex:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMComplexValue>(*value));
        break;
    case ClassType::FilterFunction:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMFilterFunctionValue>(*value));
        break;
    case ClassType::Primitive:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMPrimitiveValue>(*value));
        break;
    case ClassType::List:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMValueList>(*value));
        break;
    case ClassType::TextShadow:
        destroyAndFree(uncheckedDowncast<DeprecatedCSSOMTextShadowValue>(*value));
        break;
    }
}

unsigned short DeprecatedCSSOMValue::cssValueType() const
{
    switch (classType()) {
    case ClassType::BoxShadow:
        return uncheckedDowncast<DeprecatedCSSOMBoxShadowValue>(*this).cssValueType();
    case ClassType::Complex:
        return uncheckedDowncast<DeprecatedCSSOMComplexValue>(*this).cssValueType();
    case ClassType::FilterFunction:
        return uncheckedDowncast<DeprecatedCSSOMFilterFunctionValue>(*this).cssValueType();
    case ClassType::Primitive:
        return uncheckedDowncast<DeprecatedCSSOMPrimitiveValue>(*this).cssValueType();
    case ClassType::List:
        return CSS_VALUE_LIST;
    case ClassType::TextShadow:
        return uncheckedDowncast<DeprecatedCSSOMTextShadowValue>(*this).cssValueType();
    }
    ASSERT_NOT_REACHED();
    return CSS_CUSTOM;
}

String DeprecatedCSSOMValue::cssText() const
{
    switch (classType()) {
    case ClassType::BoxShadow:
        return uncheckedDowncast<DeprecatedCSSOMBoxShadowValue>(*this).cssText();
    case ClassType::Complex:
        return uncheckedDowncast<DeprecatedCSSOMComplexValue>(*this).cssText();
    case ClassType::FilterFunction:
        return uncheckedDowncast<DeprecatedCSSOMFilterFunctionValue>(*this).cssText();
    case ClassType::Primitive:
        return uncheckedDowncast<DeprecatedCSSOMPrimitiveValue>(*this).cssText();
    case ClassType::List:
        return uncheckedDowncast<DeprecatedCSSOMValueList>(*this).cssText();
    case ClassType::TextShadow:
        return uncheckedDowncast<DeprecatedCSSOMTextShadowValue>(*this).cssText();
    }
    ASSERT_NOT_REACHED();
    return emptyString();
}

unsigned short DeprecatedCSSOMComplexValue::cssValueType() const
{
    // These values are exposed in the DOM, but constants for them are not.
    constexpr unsigned short CSS_INITIAL = 4;
    constexpr unsigned short CSS_UNSET = 5;
    constexpr unsigned short CSS_REVERT = 6;
    switch (valueID(m_value.get())) {
    case CSSValueInherit:
        return CSS_INHERIT;
    case CSSValueInitial:
        return CSS_INITIAL;
    case CSSValueUnset:
        return CSS_UNSET;
    case CSSValueRevert:
        return CSS_REVERT;
    default:
        return CSS_CUSTOM;
    }
}

}
