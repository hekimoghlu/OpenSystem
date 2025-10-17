/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

#include "CSSStyleDeclaration.h"
#include "CSSValue.h"
#include "ExceptionOr.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DeprecatedCSSOMValue : public RefCountedAndCanMakeWeakPtr<DeprecatedCSSOMValue> {
public:
    // Exactly match the IDL. No reason to add anything if it's not in the IDL.
    enum Type : unsigned short {
        CSS_INHERIT = 0,
        CSS_PRIMITIVE_VALUE = 1,
        CSS_VALUE_LIST = 2,
        CSS_CUSTOM = 3
    };

    WEBCORE_EXPORT unsigned short cssValueType() const;

    WEBCORE_EXPORT String cssText() const;
    ExceptionOr<void> setCssText(const String&) { return { }; } // Will never implement.

    bool isBoxShadowValue() const { return classType() == ClassType::BoxShadow; }
    bool isComplexValue() const { return classType() == ClassType::Complex; }
    bool isFilterFunctionValue() const { return classType() == ClassType::FilterFunction; }
    bool isPrimitiveValue() const { return classType() == ClassType::Primitive; }
    bool isTextShadowValue() const { return classType() == ClassType::TextShadow; }
    bool isValueList() const { return classType() == ClassType::List; }

    CSSStyleDeclaration& owner() const { return m_owner; }

    // NOTE: This destructor is non-virtual for memory and performance reasons.
    // Don't go making it virtual again unless you know exactly what you're doing!
    ~DeprecatedCSSOMValue() = default;
    WEBCORE_EXPORT void operator delete(DeprecatedCSSOMValue*, std::destroying_delete_t);

protected:
    static const size_t ClassTypeBits = 3;
    enum class ClassType : uint8_t {
        BoxShadow,
        Complex,
        FilterFunction,
        List,
        Primitive,
        TextShadow
    };
    ClassType classType() const { return static_cast<ClassType>(m_classType); }

    DeprecatedCSSOMValue(ClassType classType, CSSStyleDeclaration& owner)
        : m_classType(enumToUnderlyingType(classType))
        , m_owner(owner)
    {
    }

protected:
    unsigned m_valueSeparator : CSSValue::ValueSeparatorBits;
    unsigned m_classType : ClassTypeBits; // ClassType
    
    Ref<CSSStyleDeclaration> m_owner;
};

class DeprecatedCSSOMComplexValue : public DeprecatedCSSOMValue {
public:
    static Ref<DeprecatedCSSOMComplexValue> create(Ref<const CSSValue> value, CSSStyleDeclaration& owner)
    {
        return adoptRef(*new DeprecatedCSSOMComplexValue(WTFMove(value), owner));
    }

    String cssText() const { return m_value->cssText(); }

    unsigned short cssValueType() const;

protected:
    DeprecatedCSSOMComplexValue(Ref<const CSSValue> value, CSSStyleDeclaration& owner)
        : DeprecatedCSSOMValue(ClassType::Complex, owner)
        , m_value(WTFMove(value))
    {
    }

private:
    Ref<const CSSValue> m_value;
};
    
} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CSSOM_VALUE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
static bool isType(const WebCore::DeprecatedCSSOMValue& value) { return value.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_CSSOM_VALUE(DeprecatedCSSOMComplexValue, isComplexValue())
