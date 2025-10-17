/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

#include "ExceptionOr.h"
#include "SVGPropertyTraits.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGNumber : public SVGValueProperty<float> {
    using Base = SVGValueProperty<float>;
    using Base::Base;
    using Base::m_value;
        
public:
    static Ref<SVGNumber> create(float value = 0)
    {
        return adoptRef(*new SVGNumber(value));
    }

    static Ref<SVGNumber> create(SVGPropertyOwner* owner, SVGPropertyAccess access, float value = 0)
    {
        return adoptRef(*new SVGNumber(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGNumber>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGNumber(value.releaseReturnValue()));
    }

    Ref<SVGNumber> clone() const
    {
        return SVGNumber::create(m_value);
    }

    float valueForBindings()
    {
        return m_value;
    }

    ExceptionOr<void> setValueForBindings(float value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value = value;
        commitChange();
        return { };
    }

    String valueAsString() const override
    {
        return SVGPropertyTraits<float>::toString(m_value);
    }
};

} // namespace WebCore
