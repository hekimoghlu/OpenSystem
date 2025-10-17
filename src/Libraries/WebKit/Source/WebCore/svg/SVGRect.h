/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#include "SVGPropertyTraits.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGRect : public SVGValueProperty<FloatRect> {
    using Base = SVGValueProperty<FloatRect>;
    using Base::Base;
    using Base::m_value;

public:
    static Ref<SVGRect> create(const FloatRect& value = { })
    {
        return adoptRef(*new SVGRect(value));
    }

    static Ref<SVGRect> create(SVGPropertyOwner* owner, SVGPropertyAccess access, const FloatRect& value = { })
    {
        return adoptRef(*new SVGRect(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGRect>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGRect(value.releaseReturnValue()));
    }

    float x() { return m_value.x(); }

    ExceptionOr<void> setX(float xValue)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setX(xValue);
        commitChange();
        return { };
    }

    float y() { return m_value.y(); }

    ExceptionOr<void> setY(float yValue)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setY(yValue);
        commitChange();
        return { };
    }

    float width() { return m_value.width(); }

    ExceptionOr<void> setWidth(float widthValue)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setWidth(widthValue);
        commitChange();
        return { };
    }

    float height() { return m_value.height(); }

    ExceptionOr<void> setHeight(float heightValue)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setHeight(heightValue);
        commitChange();
        return { };
    }
    
    String valueAsString() const override
    {
        return SVGPropertyTraits<FloatRect>::toString(m_value);
    }
};

}
