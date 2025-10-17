/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#include "SVGAnimationDiscreteFunction.h"
#include "SVGPropertyTraits.h"

namespace WebCore {

class SVGAnimationBooleanFunction : public SVGAnimationDiscreteFunction<bool> {
public:
    using Base = SVGAnimationDiscreteFunction<bool>;
    using Base::Base;

    void setFromAndToValues(SVGElement&, const String& from, const String& to) override
    {
        m_from = SVGPropertyTraits<bool>::fromString(from);
        m_to = SVGPropertyTraits<bool>::fromString(to);
    }
};

template<typename EnumType>
class SVGAnimationEnumerationFunction : public SVGAnimationDiscreteFunction<EnumType> {
    using Base = SVGAnimationDiscreteFunction<EnumType>;
    using Base::m_from;
    using Base::m_to;

public:
    using Base::Base;

    void setFromAndToValues(SVGElement&, const String& from, const String& to) override
    {
        m_from = SVGPropertyTraits<EnumType>::fromString(from);
        m_to = SVGPropertyTraits<EnumType>::fromString(to);
    }
};

class SVGAnimationOrientTypeFunction : public SVGAnimationDiscreteFunction<SVGMarkerOrientType> {
public:
    using Base = SVGAnimationDiscreteFunction<SVGMarkerOrientType>;
    using Base::Base;

    void setFromAndToValues(SVGElement&, const String&, const String&) override
    {
        // Values will be set by SVGAnimatedAngleOrientAnimator.
        ASSERT_NOT_REACHED();
    }

private:
    friend class SVGAnimatedAngleOrientAnimator;
};

class SVGAnimationPreserveAspectRatioFunction : public SVGAnimationDiscreteFunction<SVGPreserveAspectRatioValue> {
public:
    using Base = SVGAnimationDiscreteFunction<SVGPreserveAspectRatioValue>;
    using Base::Base;
    
    void setFromAndToValues(SVGElement&, const String& from, const String& to) override
    {
        m_from = SVGPreserveAspectRatioValue(from);
        m_to = SVGPreserveAspectRatioValue(to);
    }
};

class SVGAnimationStringFunction : public SVGAnimationDiscreteFunction<String> {
public:
    using Base = SVGAnimationDiscreteFunction<String>;
    using Base::Base;
    
    void setFromAndToValues(SVGElement&, const String& from, const String& to) override
    {
        m_from = from;
        m_to = to;
    }
};

} // namespace WebCore
