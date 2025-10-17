/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include "CSSPropertyNames.h"
#include "CSSValue.h"
#include "ScriptWrappable.h"
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename T> class ExceptionOr;
struct CSSParserContext;
class Document;

enum class CSSStyleValueType : uint8_t {
    CSSStyleValue,
    CSSStyleImageValue,
    CSSTransformValue,
    CSSMathClamp,
    CSSMathInvert,
    CSSMathMin,
    CSSMathMax,
    CSSMathNegate,
    CSSMathProduct,
    CSSMathSum,
    CSSUnitValue,
    CSSUnparsedValue,
    CSSKeywordValue
};

inline bool isCSSNumericValue(CSSStyleValueType type)
{
    switch (type) {
    case CSSStyleValueType::CSSMathClamp:
    case CSSStyleValueType::CSSMathInvert:
    case CSSStyleValueType::CSSMathMin:
    case CSSStyleValueType::CSSMathMax:
    case CSSStyleValueType::CSSMathNegate:
    case CSSStyleValueType::CSSMathProduct:
    case CSSStyleValueType::CSSMathSum:
    case CSSStyleValueType::CSSUnitValue:
        return true;
    case CSSStyleValueType::CSSStyleValue:
    case CSSStyleValueType::CSSStyleImageValue:
    case CSSStyleValueType::CSSTransformValue:
    case CSSStyleValueType::CSSUnparsedValue:
    case CSSStyleValueType::CSSKeywordValue:
        break;
    }
    return false;
}

inline bool isCSSMathValue(CSSStyleValueType type)
{
    switch (type) {
    case CSSStyleValueType::CSSMathClamp:
    case CSSStyleValueType::CSSMathInvert:
    case CSSStyleValueType::CSSMathMin:
    case CSSStyleValueType::CSSMathMax:
    case CSSStyleValueType::CSSMathNegate:
    case CSSStyleValueType::CSSMathProduct:
    case CSSStyleValueType::CSSMathSum:
        return true;
    case CSSStyleValueType::CSSUnitValue:
    case CSSStyleValueType::CSSStyleValue:
    case CSSStyleValueType::CSSStyleImageValue:
    case CSSStyleValueType::CSSTransformValue:
    case CSSStyleValueType::CSSUnparsedValue:
    case CSSStyleValueType::CSSKeywordValue:
        break;
    }
    return false;
}

enum class SerializationArguments : uint8_t {
    Nested = 0x1,
    WithoutParentheses = 0x2,
};

class CSSStyleValue : public RefCounted<CSSStyleValue>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSStyleValue);
public:
    String toString() const;
    virtual void serialize(StringBuilder&, OptionSet<SerializationArguments> = { }) const;

IGNORE_GCC_WARNINGS_BEGIN("mismatched-new-delete")
    // https://webkit.org/b/241516
    virtual ~CSSStyleValue() = default;
IGNORE_GCC_WARNINGS_END

    virtual CSSStyleValueType getType() const { return CSSStyleValueType::CSSStyleValue; }

    static ExceptionOr<Ref<CSSStyleValue>> parse(const Document&, const AtomString&, const String&);
    static ExceptionOr<Vector<Ref<CSSStyleValue>>> parseAll(const Document&, const AtomString&, const String&);

    static Ref<CSSStyleValue> create(RefPtr<CSSValue>&&, String&& = String());
    static Ref<CSSStyleValue> create();

    virtual RefPtr<CSSValue> toCSSValue() const { return m_propertyValue; }
    virtual RefPtr<CSSValue> toCSSValueWithProperty(CSSPropertyID) const { return toCSSValue(); }

protected:
    CSSStyleValue(RefPtr<CSSValue>&&, String&& = String());
    CSSStyleValue() = default;
    
    String m_customPropertyName;
    RefPtr<CSSValue> m_propertyValue;
};

} // namespace WebCore
