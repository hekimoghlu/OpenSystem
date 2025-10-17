/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "SVGProperty.h"
#include <wtf/EnumTraits.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class SVGPathSegType : uint8_t {
    Unknown = 0,
    ClosePath = 1,
    MoveToAbs = 2,
    MoveToRel = 3,
    LineToAbs = 4,
    LineToRel = 5,
    CurveToCubicAbs = 6,
    CurveToCubicRel = 7,
    CurveToQuadraticAbs = 8,
    CurveToQuadraticRel = 9,
    ArcAbs = 10,
    ArcRel = 11,
    LineToHorizontalAbs = 12,
    LineToHorizontalRel = 13,
    LineToVerticalAbs = 14,
    LineToVerticalRel = 15,
    CurveToCubicSmoothAbs = 16,
    CurveToCubicSmoothRel = 17,
    CurveToQuadraticSmoothAbs = 18,
    CurveToQuadraticSmoothRel = 19
};

class SVGPathSeg : public SVGProperty {
public:
    virtual ~SVGPathSeg() = default;

    // Forward declare these enums in the w3c naming scheme, for IDL generation
    enum {
        PATHSEG_UNKNOWN = enumToUnderlyingType(SVGPathSegType::Unknown),
        PATHSEG_CLOSEPATH = enumToUnderlyingType(SVGPathSegType::ClosePath),
        PATHSEG_MOVETO_ABS = enumToUnderlyingType(SVGPathSegType::MoveToAbs),
        PATHSEG_MOVETO_REL = enumToUnderlyingType(SVGPathSegType::MoveToRel),
        PATHSEG_LINETO_ABS = enumToUnderlyingType(SVGPathSegType::LineToAbs),
        PATHSEG_LINETO_REL = enumToUnderlyingType(SVGPathSegType::LineToRel),
        PATHSEG_CURVETO_CUBIC_ABS = enumToUnderlyingType(SVGPathSegType::CurveToCubicAbs),
        PATHSEG_CURVETO_CUBIC_REL = enumToUnderlyingType(SVGPathSegType::CurveToCubicRel),
        PATHSEG_CURVETO_QUADRATIC_ABS = enumToUnderlyingType(SVGPathSegType::CurveToQuadraticAbs),
        PATHSEG_CURVETO_QUADRATIC_REL = enumToUnderlyingType(SVGPathSegType::CurveToQuadraticRel),
        PATHSEG_ARC_ABS = enumToUnderlyingType(SVGPathSegType::ArcAbs),
        PATHSEG_ARC_REL = enumToUnderlyingType(SVGPathSegType::ArcRel),
        PATHSEG_LINETO_HORIZONTAL_ABS = enumToUnderlyingType(SVGPathSegType::LineToHorizontalAbs),
        PATHSEG_LINETO_HORIZONTAL_REL = enumToUnderlyingType(SVGPathSegType::LineToHorizontalRel),
        PATHSEG_LINETO_VERTICAL_ABS = enumToUnderlyingType(SVGPathSegType::LineToVerticalAbs),
        PATHSEG_LINETO_VERTICAL_REL = enumToUnderlyingType(SVGPathSegType::LineToVerticalRel),
        PATHSEG_CURVETO_CUBIC_SMOOTH_ABS = enumToUnderlyingType(SVGPathSegType::CurveToCubicSmoothAbs),
        PATHSEG_CURVETO_CUBIC_SMOOTH_REL = enumToUnderlyingType(SVGPathSegType::CurveToCubicSmoothRel),
        PATHSEG_CURVETO_QUADRATIC_SMOOTH_ABS = enumToUnderlyingType(SVGPathSegType::CurveToQuadraticSmoothAbs),
        PATHSEG_CURVETO_QUADRATIC_SMOOTH_REL = enumToUnderlyingType(SVGPathSegType::CurveToQuadraticSmoothRel)
    };

    virtual SVGPathSegType pathSegType() const = 0;
    unsigned short pathSegTypeForBindings() const { return static_cast<unsigned short>(pathSegType()); }
    virtual String pathSegTypeAsLetter() const = 0;
    virtual Ref<SVGPathSeg> clone() const = 0;

protected:
    using SVGProperty::SVGProperty;
};

} // namespace WebCore
