/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

// <'border-radius'> = <length-percentage [0,âˆž]>{1,4} [ / <length-percentage [0,âˆž]>{1,4} ]?
// https://drafts.csswg.org/css-backgrounds-3/#propdef-border-radius
struct BorderRadius {
    using Axis = SpaceSeparatedArray<LengthPercentage<Nonnegative>, 4>;
    using Corner = SpaceSeparatedSize<LengthPercentage<Nonnegative>>;

    static BorderRadius defaultValue();

    Axis horizontal;
    Axis vertical;

    Corner topLeft() const      { return { horizontal.value[0], vertical.value[0] }; }
    Corner topRight() const     { return { horizontal.value[1], vertical.value[1] }; }
    Corner bottomRight() const  { return { horizontal.value[2], vertical.value[2] }; }
    Corner bottomLeft() const   { return { horizontal.value[3], vertical.value[3] }; }

    bool operator==(const BorderRadius&) const = default;
};

template<size_t I> const auto& get(const BorderRadius& value)
{
    if constexpr (!I)
        return value.horizontal;
    else if constexpr (I == 1)
        return value.vertical;
}

// Returns true if the provided `BorderRadius` contains the default value. This
// is used to know ahead of time if serialization is needed.
bool hasDefaultValue(const BorderRadius&);

inline BorderRadius BorderRadius::defaultValue()
{
    return BorderRadius {
        .horizontal = { 0_css_px, 0_css_px, 0_css_px, 0_css_px },
        .vertical   = { 0_css_px, 0_css_px, 0_css_px, 0_css_px },
    };
}

template<> struct Serialize<BorderRadius> { void operator()(StringBuilder&, const BorderRadius&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::BorderRadius, 2)
