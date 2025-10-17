/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#include "SVGPathSeg.h"
#include "SVGPathSegValue.h"

namespace WebCore {

class SVGPathSegClosePath final : public SVGPathSeg {
public:
    static Ref<SVGPathSegClosePath> create() { return adoptRef(*new SVGPathSegClosePath()); }
private:
    using SVGPathSeg::SVGPathSeg;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::ClosePath; }
    String pathSegTypeAsLetter() const final { return "Z"_s; }
    Ref<SVGPathSeg> clone() const final { return adoptRef(*new SVGPathSegClosePath()); }
};

class SVGPathSegLinetoHorizontalAbs final : public SVGPathSegLinetoHorizontal {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoHorizontalAbs>;
private:
    using SVGPathSegLinetoHorizontal::SVGPathSegLinetoHorizontal;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToHorizontalAbs; }
    String pathSegTypeAsLetter() const final { return "H"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoHorizontalAbs>(); }
};

class SVGPathSegLinetoHorizontalRel final : public SVGPathSegLinetoHorizontal {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoHorizontalRel>;
private:
    using SVGPathSegLinetoHorizontal::SVGPathSegLinetoHorizontal;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToHorizontalRel; }
    String pathSegTypeAsLetter() const final { return "h"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoHorizontalRel>(); }
};

class SVGPathSegLinetoVerticalAbs final : public SVGPathSegLinetoVertical {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoVerticalAbs>;
private:
    using SVGPathSegLinetoVertical::SVGPathSegLinetoVertical;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToVerticalAbs; }
    String pathSegTypeAsLetter() const final { return "V"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoVerticalAbs>(); }
};

class SVGPathSegLinetoVerticalRel final : public SVGPathSegLinetoVertical {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoVerticalRel>;
private:
    using SVGPathSegLinetoVertical::SVGPathSegLinetoVertical;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToVerticalRel; }
    String pathSegTypeAsLetter() const final { return "v"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoVerticalRel>(); }
};

class SVGPathSegMovetoAbs final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegMovetoAbs>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::MoveToAbs; }
    String pathSegTypeAsLetter() const final { return "M"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegMovetoAbs>(); }
};

class SVGPathSegMovetoRel final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegMovetoRel>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::MoveToRel; }
    String pathSegTypeAsLetter() const final { return "m"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegMovetoRel>(); }
};

class SVGPathSegLinetoAbs final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoAbs>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToAbs; }
    String pathSegTypeAsLetter() const final { return "L"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoAbs>(); }
};

class SVGPathSegLinetoRel final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegLinetoRel>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::LineToRel; }
    String pathSegTypeAsLetter() const final { return "l"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegLinetoRel>(); }
};

class SVGPathSegCurvetoQuadraticAbs final : public SVGPathSegCurvetoQuadratic {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoQuadraticAbs>;
private:
    using SVGPathSegCurvetoQuadratic::SVGPathSegCurvetoQuadratic;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToQuadraticAbs; }
    String pathSegTypeAsLetter() const final { return "Q"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoQuadraticAbs>(); }
};

class SVGPathSegCurvetoQuadraticRel final : public SVGPathSegCurvetoQuadratic {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoQuadraticRel>;
private:
    using SVGPathSegCurvetoQuadratic::SVGPathSegCurvetoQuadratic;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToQuadraticRel; }
    String pathSegTypeAsLetter() const final { return "q"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoQuadraticRel>(); }
};

class SVGPathSegCurvetoCubicAbs final : public SVGPathSegCurvetoCubic {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoCubicAbs>;
private:
    using SVGPathSegCurvetoCubic::SVGPathSegCurvetoCubic;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToCubicAbs; }
    String pathSegTypeAsLetter() const final { return "C"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoCubicAbs>(); }
};

class SVGPathSegCurvetoCubicRel final : public SVGPathSegCurvetoCubic {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoCubicRel>;
private:
    using SVGPathSegCurvetoCubic::SVGPathSegCurvetoCubic;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToCubicRel; }
    String pathSegTypeAsLetter() const final { return "c"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoCubicRel>(); }
};

class SVGPathSegArcAbs final : public SVGPathSegArc {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegArcAbs>;
private:
    using SVGPathSegArc::SVGPathSegArc;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::ArcAbs; }
    String pathSegTypeAsLetter() const final { return "A"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegArcAbs>(); }
};

class SVGPathSegArcRel final : public SVGPathSegArc {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegArcRel>;
private:
    using SVGPathSegArc::SVGPathSegArc;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::ArcRel; }
    String pathSegTypeAsLetter() const final { return "a"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegArcRel>(); }
};

class SVGPathSegCurvetoQuadraticSmoothAbs final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoQuadraticSmoothAbs>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToQuadraticSmoothAbs; }
    String pathSegTypeAsLetter() const final { return "T"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoQuadraticSmoothAbs>(); }
};

class SVGPathSegCurvetoQuadraticSmoothRel final : public SVGPathSegSingleCoordinate {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoQuadraticSmoothRel>;
private:
    using SVGPathSegSingleCoordinate::SVGPathSegSingleCoordinate;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToQuadraticSmoothRel; }
    String pathSegTypeAsLetter() const final { return "t"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoQuadraticSmoothRel>(); }
};

class SVGPathSegCurvetoCubicSmoothAbs final : public SVGPathSegCurvetoCubicSmooth {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoCubicSmoothAbs>;
private:
    using SVGPathSegCurvetoCubicSmooth::SVGPathSegCurvetoCubicSmooth;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToCubicSmoothAbs; }
    String pathSegTypeAsLetter() const final { return "S"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoCubicSmoothAbs>(); }
};

class SVGPathSegCurvetoCubicSmoothRel final : public SVGPathSegCurvetoCubicSmooth {
public:
    static constexpr auto create = SVGPathSegValue::create<SVGPathSegCurvetoCubicSmoothRel>;
private:
    using SVGPathSegCurvetoCubicSmooth::SVGPathSegCurvetoCubicSmooth;
    SVGPathSegType pathSegType() const final { return SVGPathSegType::CurveToCubicSmoothRel; }
    String pathSegTypeAsLetter() const final { return "s"_s; }
    Ref<SVGPathSeg> clone() const final { return SVGPathSegValue::cloneInternal<SVGPathSegCurvetoCubicSmoothRel>(); }
};

} // namespace WebCore
