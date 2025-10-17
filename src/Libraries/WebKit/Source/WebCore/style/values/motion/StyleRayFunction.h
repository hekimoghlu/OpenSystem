/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "CSSRayFunction.h"
#include "StylePosition.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

using RaySize = CSS::RaySize;

// ray() = ray( <angle> && <ray-size>? && contain? && [at <position>]? )
// <ray-size> = closest-side | closest-corner | farthest-side | farthest-corner | sides
// https://drafts.fxtf.org/motion-1/#ray-function
struct Ray {
    Angle<> angle;
    RaySize size;
    std::optional<CSS::Keyword::Contain> contain;
    std::optional<Position> position;

    bool operator==(const Ray&) const = default;
};
using RayFunction = FunctionNotation<CSSValueRay, Ray>;

template<size_t I> const auto& get(const Ray& value)
{
    if constexpr (!I)
        return value.angle;
    else if constexpr (I == 1)
        return value.size;
    else if constexpr (I == 2)
        return value.contain;
    else if constexpr (I == 3)
        return value.position;
}

DEFINE_TYPE_MAPPING(CSS::Ray, Ray)

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Ray, 4)
