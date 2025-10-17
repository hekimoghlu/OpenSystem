/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

#include "AffineTransform.h"
#include "ExceptionOr.h"
#include "SVGValueProperty.h"

namespace WebCore {

// FIXME: Remove this class once SVGMatrix becomes an alias to DOMMatrix.
class SVGMatrix : public SVGValueProperty<AffineTransform> {
    using Base = SVGValueProperty<AffineTransform>;
    using Base::Base;

public:
    static Ref<SVGMatrix> create(const AffineTransform& value = { })
    {
        return adoptRef(*new SVGMatrix(value));
    }

    static Ref<SVGMatrix> create(SVGPropertyOwner* owner, SVGPropertyAccess access, const AffineTransform& value = { })
    {
        return adoptRef(*new SVGMatrix(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGMatrix>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return create(value.releaseReturnValue());
    }

    double a() const
    {
        return m_value.a();
    }

    ExceptionOr<void> setA(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setA(value);
        commitChange();
        return { };
    }

    double b() const
    {
        return m_value.b();
    }

    ExceptionOr<void> setB(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setB(value);
        commitChange();
        return { };
    }

    double c() const
    {
        return m_value.c();
    }

    ExceptionOr<void> setC(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setC(value);
        commitChange();
        return { };
    }

    double d() const
    {
        return m_value.d();
    }

    ExceptionOr<void> setD(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setD(value);
        commitChange();
        return { };
    }

    double e() const
    {
        return m_value.e();
    }

    ExceptionOr<void> setE(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setE(value);
        commitChange();
        return { };
    }

    double f() const
    {
        return m_value.f();
    }

    ExceptionOr<void> setF(double value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setF(value);
        commitChange();
        return { };
    }

    Ref<SVGMatrix> multiply(SVGMatrix& secondMatrix) const
    {
        auto copy = m_value;
        copy.multiply(secondMatrix.value());
        return SVGMatrix::create(copy);
    }

    ExceptionOr<Ref<SVGMatrix>> inverse() const
    {
        auto inverse = m_value.inverse();
        if (!inverse)
            return Exception { ExceptionCode::InvalidStateError, "Matrix is not invertible"_s };
        return SVGMatrix::create(*inverse);
    }

    Ref<SVGMatrix> translate(float x, float y) const
    {
        auto copy = m_value;
        copy.translate(x, y);
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> scale(float scaleFactor) const
    {
        auto copy = m_value;
        copy.scale(scaleFactor);
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> scaleNonUniform(float scaleFactorX, float scaleFactorY) const
    {
        auto copy = m_value;
        copy.scaleNonUniform(scaleFactorX, scaleFactorY);
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> rotate(float angle) const
    {
        auto copy = m_value;
        copy.rotate(angle);
        return SVGMatrix::create(copy);
    }

    ExceptionOr<Ref<SVGMatrix>> rotateFromVector(float x, float y) const
    {
        if (!x || !y)
            return Exception { ExceptionCode::TypeError };

        auto copy = m_value;
        copy.rotateFromVector(x, y);
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> flipX() const
    {
        auto copy = m_value;
        copy.flipX();
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> flipY() const
    {
        auto copy = m_value;
        copy.flipY();
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> skewX(float angle) const
    {
        auto copy = m_value;
        copy.skewX(angle);
        return SVGMatrix::create(copy);
    }

    Ref<SVGMatrix> skewY(float angle) const
    {
        auto copy = m_value;
        copy.skewY(angle);
        return SVGMatrix::create(copy);
    }
};

} // namespace WebCore
