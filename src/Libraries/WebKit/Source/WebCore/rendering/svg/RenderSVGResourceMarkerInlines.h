/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#include "RenderSVGResourceMarker.h"
#include "SVGMarkerElement.h"

namespace WebCore {

inline SVGMarkerElement& RenderSVGResourceMarker::markerElement() const
{
    return downcast<SVGMarkerElement>(RenderSVGResourceContainer::element());
}

inline Ref<SVGMarkerElement> RenderSVGResourceMarker::protectedMarkerElement() const
{
    return markerElement();
}

FloatPoint RenderSVGResourceMarker::referencePoint() const
{
    Ref markerElement = this->markerElement();
    SVGLengthContext lengthContext(markerElement.ptr());
    return { markerElement->refX().value(lengthContext), markerElement->refY().value(lengthContext) };
}

std::optional<float> RenderSVGResourceMarker::angle() const
{
    if (Ref markerElement = this->markerElement(); markerElement->orientType() == SVGMarkerOrientAngle)
        return markerElement->orientAngle().value();
    return std::nullopt;
}

SVGMarkerUnitsType RenderSVGResourceMarker::markerUnits() const
{
    return protectedMarkerElement()->markerUnits();
}

bool RenderSVGResourceMarker::hasReverseStart() const
{
    return protectedMarkerElement()->orientType() == SVGMarkerOrientAutoStartReverse;
}

}
