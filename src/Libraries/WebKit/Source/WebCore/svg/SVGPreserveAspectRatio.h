/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#include "SVGPreserveAspectRatioValue.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGPreserveAspectRatio : public SVGValueProperty<SVGPreserveAspectRatioValue> {
    using Base = SVGValueProperty<SVGPreserveAspectRatioValue>;
    using Base::Base;
    using Base::m_value;

public:
    static Ref<SVGPreserveAspectRatio> create(SVGPropertyOwner* owner, SVGPropertyAccess access, const SVGPreserveAspectRatioValue& value = { })
    {
        return adoptRef(*new SVGPreserveAspectRatio(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGPreserveAspectRatio>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGPreserveAspectRatio(value.releaseReturnValue()));
    }

    unsigned short align() const { return m_value.align(); }

    ExceptionOr<void> setAlign(float value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        auto result = m_value.setAlign(value);
        if (result.hasException())
            return result;

        commitChange();
        return result;
    }

    unsigned short meetOrSlice() const { return m_value.meetOrSlice(); }

    ExceptionOr<void> setMeetOrSlice(float value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        auto result = m_value.setMeetOrSlice(value);
        if (result.hasException())
            return result;

        commitChange();
        return result;
    }

    String valueAsString() const override
    {
        return m_value.valueAsString();
    }
};

} // namespace WebCore
