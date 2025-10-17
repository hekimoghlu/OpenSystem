/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#include "CSSSkewX.h"

#include "CSSFunctionValue.h"
#include "CSSNumericFactory.h"
#include "CSSNumericValue.h"
#include "CSSStyleValueFactory.h"
#include "DOMMatrix.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSSkewX);

ExceptionOr<Ref<CSSSkewX>> CSSSkewX::create(Ref<CSSNumericValue> ax)
{
    if (!ax->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };
    return adoptRef(*new CSSSkewX(WTFMove(ax)));
}

ExceptionOr<Ref<CSSSkewX>> CSSSkewX::create(Ref<const CSSFunctionValue> cssFunctionValue)
{
    if (cssFunctionValue->name() != CSSValueSkewX) {
        ASSERT_NOT_REACHED();
        return CSSSkewX::create(Ref<CSSNumericValue>(CSSNumericFactory::deg(0)));
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
    return CSSSkewX::create(numericValue.releaseNonNull());
}

CSSSkewX::CSSSkewX(Ref<CSSNumericValue> ax)
    : CSSTransformComponent(Is2D::Yes)
    , m_ax(WTFMove(ax))
{
}

ExceptionOr<void> CSSSkewX::setAx(Ref<CSSNumericValue> ax)
{
    if (!ax->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };

    m_ax = WTFMove(ax);
    return { };
}

void CSSSkewX::serialize(StringBuilder& builder) const
{
    // https://drafts.css-houdini.org/css-typed-om/#serialize-a-cssskewx
    builder.append("skewX("_s);
    m_ax->serialize(builder);
    builder.append(')');
}

ExceptionOr<Ref<DOMMatrix>> CSSSkewX::toMatrix()
{
    RefPtr ax = dynamicDowncast<CSSUnitValue>(m_ax);
    if (!ax)
        return Exception { ExceptionCode::TypeError };

    auto x = ax->convertTo(CSSUnitType::CSS_DEG);
    if (!x)
        return Exception { ExceptionCode::TypeError };

    TransformationMatrix matrix { };
    matrix.skewX(x->value());

    return { DOMMatrix::create(WTFMove(matrix), DOMMatrixReadOnly::Is2D::Yes) };
}

RefPtr<CSSValue> CSSSkewX::toCSSValue() const
{
    auto ax = m_ax->toCSSValue();
    if (!ax)
        return nullptr;
    return CSSFunctionValue::create(CSSValueSkewX, ax.releaseNonNull());
}

} // namespace WebCore
