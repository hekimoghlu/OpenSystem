/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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

namespace WebCore {

template<class... Arguments>
class SVGPathSegValue : public SVGPathSeg {
public:
    template<typename PathSegType>
    static Ref<PathSegType> create(Arguments... arguments)
    {
        return adoptRef(*new PathSegType(std::forward<Arguments>(arguments)...));
    }

    SVGPathSegValue(Arguments... arguments)
        : m_arguments(std::forward<Arguments>(arguments)...)
    {
    }

    SVGPathSegValue(const std::tuple<Arguments...>& arguments)
        : m_arguments(arguments)
    {
    }

protected:
    template<size_t I>
    const auto& argument() const
    {
        return std::get<I>(m_arguments);
    }

    template<size_t I, typename ArgumentValue>
    void setArgument(ArgumentValue value)
    {
        std::get<I>(m_arguments) = value;
        commitChange();
    }

    template<typename PathSegType>
    Ref<PathSegType> cloneInternal() const
    {
        return adoptRef(*new PathSegType(m_arguments));
    }

    std::tuple<Arguments...> m_arguments;
};

class SVGPathSegLinetoHorizontal : public SVGPathSegValue<float> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegLinetoVertical : public SVGPathSegValue<float> {
public:
    float y() const { return argument<0>(); }
    void setY(float x) { setArgument<0>(x); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegSingleCoordinate : public SVGPathSegValue<float, float> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

    float y() const { return argument<1>(); }
    void setY(float y) { setArgument<1>(y); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegCurvetoQuadratic : public SVGPathSegValue<float, float, float, float> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

    float y() const { return argument<1>(); }
    void setY(float y) { setArgument<1>(y); }

    float x1() const { return argument<2>(); }
    void setX1(float x) { setArgument<2>(x); }

    float y1() const { return argument<3>(); }
    void setY1(float y) { setArgument<3>(y); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegCurvetoCubicSmooth : public SVGPathSegValue<float, float, float, float> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

    float y() const { return argument<1>(); }
    void setY(float y) { setArgument<1>(y); }

    float x2() const { return argument<2>(); }
    void setX2(float x) { setArgument<2>(x); }

    float y2() const { return argument<3>(); }
    void setY2(float y) { setArgument<3>(y); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegCurvetoCubic : public SVGPathSegValue<float, float, float, float, float, float> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

    float y() const { return argument<1>(); }
    void setY(float y) { setArgument<1>(y); }

    float x1() const { return argument<2>(); }
    void setX1(float x) { setArgument<2>(x); }

    float y1() const { return argument<3>(); }
    void setY1(float y) { setArgument<3>(y); }

    float x2() const { return argument<4>(); }
    void setX2(float x) { setArgument<4>(x); }

    float y2() const { return argument<5>(); }
    void setY2(float y) { setArgument<5>(y); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

class SVGPathSegArc : public SVGPathSegValue<float, float, float, float, float, bool, bool> {
public:
    float x() const { return argument<0>(); }
    void setX(float x) { setArgument<0>(x); }

    float y() const { return argument<1>(); }
    void setY(float y) { setArgument<1>(y); }

    float r1() const { return argument<2>(); }
    void setR1(float r1) { setArgument<2>(r1); }

    float r2() const { return argument<3>(); }
    void setR2(float r2) { setArgument<3>(r2); }

    float angle() const { return argument<4>(); }
    void setAngle(float angle) { setArgument<4>(angle); }

    bool largeArcFlag() const { return argument<5>(); }
    void setLargeArcFlag(bool largeArcFlag) { setArgument<5>(largeArcFlag); }

    bool sweepFlag() const { return argument<6>(); }
    void setSweepFlag(bool sweepFlag) { setArgument<6>(sweepFlag); }

private:
    using SVGPathSegValue::SVGPathSegValue;
};

} // namespace WebCore
