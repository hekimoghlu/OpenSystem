/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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

#include "SVGAngle.h"
#include "SVGAnimatedDecoratedProperty.h"
#include "SVGAnimatedPrimitiveProperty.h"
#include "SVGAnimatedPropertyList.h"
#include "SVGAnimatedString.h"
#include "SVGAnimatedValueProperty.h"
#include "SVGDecoratedEnumeration.h"
#include "SVGLength.h"
#include "SVGLengthList.h"
#include "SVGMarkerTypes.h"
#include "SVGNumberList.h"
#include "SVGPathSegList.h"
#include "SVGPointList.h"
#include "SVGPreserveAspectRatio.h"
#include "SVGRect.h"
#include "SVGTransformList.h"

namespace WebCore {

using SVGAnimatedBoolean = SVGAnimatedPrimitiveProperty<bool>;
using SVGAnimatedInteger = SVGAnimatedPrimitiveProperty<int>;
using SVGAnimatedNumber = SVGAnimatedPrimitiveProperty<float>;

using SVGAnimatedEnumeration = SVGAnimatedDecoratedProperty<SVGDecoratedEnumeration, unsigned>;

using SVGAnimatedAngle = SVGAnimatedValueProperty<SVGAngle>;
using SVGAnimatedLength = SVGAnimatedValueProperty<SVGLength>;
using SVGAnimatedRect = SVGAnimatedValueProperty<SVGRect>;
using SVGAnimatedPreserveAspectRatio = SVGAnimatedValueProperty<SVGPreserveAspectRatio>;

using SVGAnimatedLengthList = SVGAnimatedPropertyList<SVGLengthList>;
using SVGAnimatedNumberList = SVGAnimatedPropertyList<SVGNumberList>;
using SVGAnimatedPointList = SVGAnimatedPropertyList<SVGPointList>;
using SVGAnimatedTransformList = SVGAnimatedPropertyList<SVGTransformList>;

class SVGAnimatedOrientType : public SVGAnimatedEnumeration {
public:
    using SVGAnimatedEnumeration::SVGAnimatedEnumeration;

    static Ref<SVGAnimatedOrientType> create(SVGElement* contextElement, SVGMarkerOrientType baseValue)
    {
        return SVGAnimatedEnumeration::create<SVGMarkerOrientType, SVGAnimatedOrientType>(contextElement, baseValue);
    }
};

class SVGAnimatedPathSegList : public SVGAnimatedPropertyList<SVGPathSegList> {
    using Base = SVGAnimatedPropertyList<SVGPathSegList>;
    using Base::Base;

public:
    static Ref<SVGAnimatedPathSegList> create(SVGElement* contextElement)
    {
        return adoptRef(*new SVGAnimatedPathSegList(contextElement));
    }

    SVGPathByteStream& currentPathByteStream()
    {
        return isAnimating() ? animVal()->pathByteStream() : baseVal()->pathByteStream();
    }

    Path currentPath()
    {
        return isAnimating() ? animVal()->path() : baseVal()->path();
    }

    size_t approximateMemoryCost() const
    {
        if (isAnimating())
            return baseVal()->approximateMemoryCost() + animVal()->approximateMemoryCost();
        return baseVal()->approximateMemoryCost();
    }
};

}
