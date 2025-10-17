/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include "CSSNumericType.h"
#include "CSSStyleValue.h"
#include <variant>
#include <wtf/HashMap.h>

namespace WebCore {

namespace CSSCalc {
struct Child;
struct ChildOrNone;
struct Tree;
}

class CSSNumericValue;
class CSSUnitValue;
class CSSMathSum;

template<typename> class ExceptionOr;

using CSSNumberish = std::variant<double, RefPtr<CSSNumericValue>>;

class CSSNumericValue : public CSSStyleValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSNumericValue);
public:

    ExceptionOr<Ref<CSSNumericValue>> add(FixedVector<CSSNumberish>&&);
    ExceptionOr<Ref<CSSNumericValue>> sub(FixedVector<CSSNumberish>&&);
    ExceptionOr<Ref<CSSNumericValue>> mul(FixedVector<CSSNumberish>&&);
    ExceptionOr<Ref<CSSNumericValue>> div(FixedVector<CSSNumberish>&&);
    ExceptionOr<Ref<CSSNumericValue>> min(FixedVector<CSSNumberish>&&);
    ExceptionOr<Ref<CSSNumericValue>> max(FixedVector<CSSNumberish>&&);
    
    bool equals(FixedVector<CSSNumberish>&&);
    
    ExceptionOr<Ref<CSSUnitValue>> to(String&&);
    ExceptionOr<Ref<CSSUnitValue>> to(CSSUnitType);
    ExceptionOr<Ref<CSSMathSum>> toSum(FixedVector<String>&&);

    const CSSNumericType& type() const { return m_type; }
    
    static ExceptionOr<Ref<CSSNumericValue>> parse(Document&, String&&);
    static Ref<CSSNumericValue> rectifyNumberish(CSSNumberish&&);

    // https://drafts.css-houdini.org/css-typed-om/#sum-value-value
    using UnitMap = UncheckedKeyHashMap<CSSUnitType, int, WTF::IntHash<CSSUnitType>, WTF::StrongEnumHashTraits<CSSUnitType>>;
    struct Addend {
        double value;
        UnitMap units;
    };
    using SumValue = Vector<Addend>;
    virtual std::optional<SumValue> toSumValue() const = 0;
    virtual bool equals(const CSSNumericValue&) const = 0;

    virtual std::optional<CSSCalc::Child> toCalcTreeNode() const = 0;

    static ExceptionOr<Ref<CSSNumericValue>> reifyMathExpression(const CSSCalc::Tree&);
    static ExceptionOr<Ref<CSSNumericValue>> reifyMathExpression(const CSSCalc::Child&);
    static ExceptionOr<Ref<CSSNumericValue>> reifyMathExpression(const CSSCalc::ChildOrNone&);

protected:
    ExceptionOr<Ref<CSSNumericValue>> addInternal(Vector<Ref<CSSNumericValue>>&&);
    ExceptionOr<Ref<CSSNumericValue>> multiplyInternal(Vector<Ref<CSSNumericValue>>&&);
    template<typename T> Vector<Ref<CSSNumericValue>> prependItemsOfTypeOrThis(Vector<Ref<CSSNumericValue>>&&);

    CSSNumericValue(CSSNumericType type = { })
        : m_type(WTFMove(type)) { }

    CSSNumericType m_type;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSNumericValue)
    static bool isType(const WebCore::CSSStyleValue& styleValue) { return isCSSNumericValue(styleValue.getType()); }
SPECIALIZE_TYPE_TRAITS_END()
