/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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

#include "FloatPoint.h"
#include "SVGMatrix.h"
#include "SVGPropertyTraits.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGPoint : public SVGValueProperty<FloatPoint> {
    using Base = SVGValueProperty<FloatPoint>;
    using Base::Base;
    using Base::m_value;

public:
    static Ref<SVGPoint> create(const FloatPoint& value = { })
    {
        return adoptRef(*new SVGPoint(value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGPoint>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGPoint(value.releaseReturnValue()));
    }

    Ref<SVGPoint> clone() const
    {
        return SVGPoint::create(m_value);
    }
    
    float x() { return m_value.x(); }

    ExceptionOr<void> setX(float x)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setX(x);
        commitChange();

        return { };
    }

    float y() { return m_value.y(); }

    ExceptionOr<void> setY(float y)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setY(y);
        commitChange();
        return { };
    }

    Ref<SVGPoint> matrixTransform(SVGMatrix& matrix) const
    {
        auto newPoint = m_value.matrixTransform(matrix.value());
        return adoptRef(*new SVGPoint(newPoint));
    }

private:
    String valueAsString() const override
    {
        return SVGPropertyTraits<FloatPoint>::toString(m_value);
    }
};

} // namespace WebCore
