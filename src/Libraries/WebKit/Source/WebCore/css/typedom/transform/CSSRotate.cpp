/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include "CSSRotate.h"

#include "CSSFunctionValue.h"
#include "CSSNumericFactory.h"
#include "CSSNumericValue.h"
#include "CSSStyleValueFactory.h"
#include "CSSUnitValue.h"
#include "CSSUnits.h"
#include "DOMMatrix.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSRotate);

ExceptionOr<Ref<CSSRotate>> CSSRotate::create(CSSNumberish x, CSSNumberish y, CSSNumberish z, Ref<CSSNumericValue> angle)
{
    if (!angle->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };

    auto rectifiedX = CSSNumericValue::rectifyNumberish(WTFMove(x));
    auto rectifiedY = CSSNumericValue::rectifyNumberish(WTFMove(y));
    auto rectifiedZ = CSSNumericValue::rectifyNumberish(WTFMove(z));

    if (!rectifiedX->type().matchesNumber()
        || !rectifiedY->type().matchesNumber()
        || !rectifiedZ->type().matchesNumber())
        return Exception { ExceptionCode::TypeError };

    return adoptRef(*new CSSRotate(Is2D::No, WTFMove(rectifiedX), WTFMove(rectifiedY), WTFMove(rectifiedZ), WTFMove(angle)));
}

ExceptionOr<Ref<CSSRotate>> CSSRotate::create(Ref<CSSNumericValue> angle)
{
    if (!angle->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };
    return adoptRef(*new CSSRotate(Is2D::Yes,
        CSSUnitValue::create(0.0, CSSUnitType::CSS_NUMBER),
        CSSUnitValue::create(0.0, CSSUnitType::CSS_NUMBER),
        CSSUnitValue::create(1.0, CSSUnitType::CSS_NUMBER),
        WTFMove(angle)));
}

ExceptionOr<Ref<CSSRotate>> CSSRotate::create(Ref<const CSSFunctionValue> cssFunctionValue)
{
    auto makeRotate = [&](const Function<ExceptionOr<Ref<CSSRotate>>(Vector<RefPtr<CSSNumericValue>>&&)>& create, size_t expectedNumberOfComponents) -> ExceptionOr<Ref<CSSRotate>> {
        Vector<RefPtr<CSSNumericValue>> components;
        for (auto& componentCSSValue : cssFunctionValue.get()) {
            auto valueOrException = CSSStyleValueFactory::reifyValue(componentCSSValue, std::nullopt);
            if (valueOrException.hasException())
                return valueOrException.releaseException();
            RefPtr numericValue = dynamicDowncast<CSSNumericValue>(valueOrException.releaseReturnValue());
            if (!numericValue)
                return Exception { ExceptionCode::TypeError, "Expected a CSSNumericValue."_s };
            components.append(WTFMove(numericValue));
        }
        if (components.size() != expectedNumberOfComponents) {
            ASSERT_NOT_REACHED();
            return Exception { ExceptionCode::TypeError, "Unexpected number of values."_s };
        }
        return create(WTFMove(components));
    };

    switch (cssFunctionValue->name()) {
    case CSSValueRotateX:
        return makeRotate([](Vector<RefPtr<CSSNumericValue>>&& components) {
            return CSSRotate::create(CSSNumericFactory::number(1), CSSNumericFactory::number(0), CSSNumericFactory::number(0), *components[0]);
        }, 1);
    case CSSValueRotateY:
        return makeRotate([](Vector<RefPtr<CSSNumericValue>>&& components) {
            return CSSRotate::create(CSSNumericFactory::number(0), CSSNumericFactory::number(1), CSSNumericFactory::number(0), *components[0]);
        }, 1);
    case CSSValueRotateZ:
        return makeRotate([](Vector<RefPtr<CSSNumericValue>>&& components) {
            return CSSRotate::create(CSSNumericFactory::number(0), CSSNumericFactory::number(0), CSSNumericFactory::number(1), *components[0]);
        }, 1);
    case CSSValueRotate:
        return makeRotate([](Vector<RefPtr<CSSNumericValue>>&& components) {
            return CSSRotate::create(*components[0]);
        }, 1);
    case CSSValueRotate3d:
        return makeRotate([](Vector<RefPtr<CSSNumericValue>>&& components) {
            return CSSRotate::create(components[0], components[1], components[2], *components[3]);
        }, 4);
    default:
        ASSERT_NOT_REACHED();
        return CSSRotate::create(Ref<CSSNumericValue>(CSSNumericFactory::deg(0)));
    }
}

CSSRotate::CSSRotate(CSSTransformComponent::Is2D is2D, Ref<CSSNumericValue> x, Ref<CSSNumericValue> y, Ref<CSSNumericValue> z, Ref<CSSNumericValue> angle)
    : CSSTransformComponent(is2D)
    , m_x(WTFMove(x))
    , m_y(WTFMove(y))
    , m_z(WTFMove(z))
    , m_angle(WTFMove(angle))
{
}

ExceptionOr<void> CSSRotate::setX(CSSNumberish x)
{
    auto rectified = CSSNumericValue::rectifyNumberish(WTFMove(x));
    if (!rectified->type().matchesNumber())
        return Exception { ExceptionCode::TypeError };
    m_x = WTFMove(rectified);
    return { };
}

ExceptionOr<void> CSSRotate::setY(CSSNumberish y)
{
    auto rectified = CSSNumericValue::rectifyNumberish(WTFMove(y));
    if (!rectified->type().matchesNumber())
        return Exception { ExceptionCode::TypeError };
    m_y = WTFMove(rectified);
    return { };
}

ExceptionOr<void> CSSRotate::setZ(CSSNumberish z)
{
    auto rectified = CSSNumericValue::rectifyNumberish(WTFMove(z));
    if (!rectified->type().matchesNumber())
        return Exception { ExceptionCode::TypeError };
    m_z = WTFMove(rectified);
    return { };
}

ExceptionOr<void> CSSRotate::setAngle(Ref<CSSNumericValue> angle)
{
    if (!angle->type().matches<CSSNumericBaseType::Angle>())
        return Exception { ExceptionCode::TypeError };
    m_angle = WTFMove(angle);
    return { };
}

void CSSRotate::serialize(StringBuilder& builder) const
{
    // https://drafts.css-houdini.org/css-typed-om/#serialize-a-cssrotate
    builder.append(is2D() ? "rotate("_s : "rotate3d("_s);
    if (!is2D()) {
        m_x->serialize(builder);
        builder.append(", "_s);
        m_y->serialize(builder);
        builder.append(", "_s);
        m_z->serialize(builder);
        builder.append(", "_s);
    }
    m_angle->serialize(builder);
    builder.append(')');
}

ExceptionOr<Ref<DOMMatrix>> CSSRotate::toMatrix()
{
    RefPtr angleUnitValue = dynamicDowncast<CSSUnitValue>(m_angle);
    RefPtr xUnitValue = dynamicDowncast<CSSUnitValue>(m_x);
    RefPtr yUnitValue = dynamicDowncast<CSSUnitValue>(m_y);
    RefPtr zUnitValue = dynamicDowncast<CSSUnitValue>(m_z);
    if (!angleUnitValue || !xUnitValue || !yUnitValue || !zUnitValue)
        return Exception { ExceptionCode::TypeError };

    auto angle = angleUnitValue->convertTo(CSSUnitType::CSS_DEG);
    if (!angle)
        return Exception { ExceptionCode::TypeError };

    TransformationMatrix matrix { };

    if (is2D())
        matrix.rotate(angle->value());
    else {
        auto x = xUnitValue->value();
        auto y = yUnitValue->value();
        auto z = zUnitValue->value();

        matrix.rotate3d(x, y, z, angle->value());
    }

    return { DOMMatrix::create(WTFMove(matrix), is2D() ? DOMMatrixReadOnly::Is2D::Yes : DOMMatrixReadOnly::Is2D::No) };
}

RefPtr<CSSValue> CSSRotate::toCSSValue() const
{
    auto angle = m_angle->toCSSValue();
    if (!angle)
        return nullptr;

    if (is2D())
        return CSSFunctionValue::create(CSSValueRotate, angle.releaseNonNull());

    auto x = m_x->toCSSValue();
    if (!x)
        return nullptr;
    auto y = m_y->toCSSValue();
    if (!y)
        return nullptr;
    auto z = m_z->toCSSValue();
    if (!z)
        return nullptr;

    return CSSFunctionValue::create(CSSValueRotate3d, x.releaseNonNull(), y.releaseNonNull(), z.releaseNonNull(), angle.releaseNonNull());
}

} // namespace WebCore
