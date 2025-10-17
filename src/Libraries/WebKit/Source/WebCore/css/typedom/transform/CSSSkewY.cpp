/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "CSSSkewY.h"

#include "CSSFunctionValue.h"
#include "CSSNumericFactory.h"
#include "CSSNumericValue.h"
#include "CSSStyleValueFactory.h"
#include "DOMMatrix.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSSkewY);

ExceptionOr<Ref<CSSSkewY>> CSSSkewY::create(Ref<CSSNumericValue> ay)
{
    if (!ay->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };
    return adoptRef(*new CSSSkewY(WTFMove(ay)));
}

ExceptionOr<Ref<CSSSkewY>> CSSSkewY::create(Ref<const CSSFunctionValue> cssFunctionValue)
{
    if (cssFunctionValue->name() != CSSValueSkewY) {
        ASSERT_NOT_REACHED();
        return CSSSkewY::create(Ref<CSSNumericValue>(CSSNumericFactory::deg(0)));
    }

    if (cssFunctionValue->size() != 1 || !cssFunctionValue->item(0)) {
        ASSERT_NOT_REACHED();
        return Exception { ExceptionCode::TypeError, "Unexpected number of values."_s };
    }

    auto valueOrException = CSSStyleValueFactory::reifyValue(*cssFunctionValue->item(0), std::nullopt);
    if (valueOrException.hasException())
        return valueOrException.releaseException();
    RefPtr numericValue = dynamicDowncast<CSSNumericValue>(valueOrException.releaseReturnValue());
    if (!numericValue)
        return Exception { ExceptionCode::TypeError, "Expected a CSSNumericValue."_s };
    return CSSSkewY::create(numericValue.releaseNonNull());
}

CSSSkewY::CSSSkewY(Ref<CSSNumericValue> ay)
    : CSSTransformComponent(Is2D::Yes)
    , m_ay(WTFMove(ay))
{
}

ExceptionOr<void> CSSSkewY::setAy(Ref<CSSNumericValue> ay)
{
    if (!ay->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };

    m_ay = WTFMove(ay);
    return { };
}

void CSSSkewY::serialize(StringBuilder& builder) const
{
    // https://drafts.css-houdini.org/css-typed-om/#serialize-a-cssskewy
    builder.append("skewY("_s);
    m_ay->serialize(builder);
    builder.append(')');
}

ExceptionOr<Ref<DOMMatrix>> CSSSkewY::toMatrix()
{
    RefPtr ay = dynamicDowncast<CSSUnitValue>(m_ay);
    if (!ay)
        return Exception { ExceptionCode::TypeError };

    auto y = ay->convertTo(CSSUnitType::CSS_DEG);
    if (!y)
        return Exception { ExceptionCode::TypeError };

    TransformationMatrix matrix { };
    matrix.skewY(y->value());

    return { DOMMatrix::create(WTFMove(matrix), DOMMatrixReadOnly::Is2D::Yes) };
}

RefPtr<CSSValue> CSSSkewY::toCSSValue() const
{
    auto ay = m_ay->toCSSValue();
    if (!ay)
        return nullptr;
    CSSValueListBuilder arguments;
    arguments.append(ay.releaseNonNull());
    return CSSFunctionValue::create(CSSValueSkewY, WTFMove(arguments));
}

} // namespace WebCore
