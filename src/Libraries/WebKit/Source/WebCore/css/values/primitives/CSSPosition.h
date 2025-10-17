/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

struct TwoComponentPositionHorizontal {
    std::variant<Keyword::Left, Keyword::Right, Keyword::Center, LengthPercentage<>> offset;

    bool operator==(const TwoComponentPositionHorizontal&) const = default;
};
DEFINE_TYPE_WRAPPER_GET(TwoComponentPositionHorizontal, offset);

struct TwoComponentPositionVertical {
    std::variant<Keyword::Top, Keyword::Bottom, Keyword::Center, LengthPercentage<>> offset;

    bool operator==(const TwoComponentPositionVertical&) const = default;
};
DEFINE_TYPE_WRAPPER_GET(TwoComponentPositionVertical, offset);

using TwoComponentPosition              = SpaceSeparatedTuple<TwoComponentPositionHorizontal, TwoComponentPositionVertical>;

using FourComponentPositionHorizontal   = SpaceSeparatedTuple<std::variant<Keyword::Left, Keyword::Right>, LengthPercentage<>>;
using FourComponentPositionVertical     = SpaceSeparatedTuple<std::variant<Keyword::Top, Keyword::Bottom>, LengthPercentage<>>;
using FourComponentPosition             = SpaceSeparatedTuple<FourComponentPositionHorizontal, FourComponentPositionVertical>;

struct Position {
    Position(TwoComponentPosition&& twoComponent)
        : value { WTFMove(twoComponent) }
    {
    }

    Position(const TwoComponentPosition& twoComponent)
        : value { twoComponent }
    {
    }

    Position(FourComponentPosition&& fourComponent)
        : value { WTFMove(fourComponent) }
    {
    }

    Position(const FourComponentPosition& fourComponent)
        : value { fourComponent }
    {
    }

    Position(std::variant<TwoComponentPosition, FourComponentPosition>&& variant)
        : value { WTFMove(variant) }
    {
    }

    template<typename... F> decltype(auto) switchOn(F&&... f) const
    {
        return WTF::switchOn(value, std::forward<F>(f)...);
    }

    bool operator==(const Position&) const = default;

    std::variant<TwoComponentPosition, FourComponentPosition> value;
};
DEFINE_TYPE_WRAPPER_GET(Position, value);

bool isCenterPosition(const Position&);

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::TwoComponentPositionHorizontal, 1)
DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::TwoComponentPositionVertical, 1)
DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Position, 1)
