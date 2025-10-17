/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#include "CSSPosition.h"
#include "FloatPoint.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

struct TwoComponentPositionHorizontal {
    LengthPercentage<> offset;

    bool operator==(const TwoComponentPositionHorizontal&) const = default;
};
DEFINE_TYPE_WRAPPER_GET(TwoComponentPositionHorizontal, offset);

struct TwoComponentPositionVertical {
    LengthPercentage<> offset;

    bool operator==(const TwoComponentPositionVertical&) const = default;
};
DEFINE_TYPE_WRAPPER_GET(TwoComponentPositionVertical, offset);

struct Position  {
    Position(TwoComponentPositionHorizontal&& x, TwoComponentPositionVertical&& y)
        : value { WTFMove(x.offset), WTFMove(y.offset) }
    {
    }

    Position(LengthPercentage<>&& x, LengthPercentage<>&& y)
        : value { WTFMove(x), WTFMove(y) }
    {
    }

    Position(SpaceSeparatedPoint<LengthPercentage<>>&& point)
        : value { WTFMove(point) }
    {
    }

    Position(FloatPoint point)
        : value { LengthPercentage<> { Length<> { point.x() } }, LengthPercentage<> { Length<> { point.y() } } }
    {
    }

    bool operator==(const Position&) const = default;

    LengthPercentage<> x() const { return value.x(); }
    LengthPercentage<> y() const { return value.y(); }

    SpaceSeparatedPoint<LengthPercentage<>> value;
};

template<size_t I> const auto& get(const Position& position)
{
    return get<I>(position.value);
}

// MARK: - Conversion

// Specialization is needed for ToStyle to implement resolution of keyword value to <length-percentage>.
template<> struct ToCSSMapping<TwoComponentPositionHorizontal> { using type = CSS::TwoComponentPositionHorizontal; };
template<> struct ToStyle<CSS::TwoComponentPositionHorizontal> { auto operator()(const CSS::TwoComponentPositionHorizontal&, const BuilderState&) -> TwoComponentPositionHorizontal; };
template<> struct ToCSSMapping<TwoComponentPositionVertical> { using type = CSS::TwoComponentPositionVertical; };
template<> struct ToStyle<CSS::TwoComponentPositionVertical> { auto operator()(const CSS::TwoComponentPositionVertical&, const BuilderState&) -> TwoComponentPositionVertical; };

// Specialization is needed for both ToCSS and ToStyle due to differences in type structure.
template<> struct ToCSS<Position> { auto operator()(const Position&, const RenderStyle&) -> CSS::Position; };
template<> struct ToStyle<CSS::Position> { auto operator()(const CSS::Position&, const BuilderState&) -> Position; };

// MARK: - Evaluation

template<> struct Evaluation<Position> { auto operator()(const Position&, FloatSize) -> FloatPoint; };

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::TwoComponentPositionHorizontal, 1)
DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::TwoComponentPositionVertical, 1)
DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Position, 2)
