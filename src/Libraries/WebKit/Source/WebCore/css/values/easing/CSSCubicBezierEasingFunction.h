/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

// <cubic-bezier()> = cubic-bezier( [ <number [0,1]>, <number> ]#{2} )
// https://drafts.csswg.org/css-easing-2/#funcdef-cubic-bezier
struct CubicBezierEasingParameters {
    using Coordinate = CommaSeparatedTuple<
        Number<ClosedUnitRange>,
        Number<>
    >;

    CommaSeparatedPair<Coordinate> value;

    bool operator==(const CubicBezierEasingParameters&) const = default;
};
using CubicBezierEasingFunction = FunctionNotation<CSSValueCubicBezier, CubicBezierEasingParameters>;

DEFINE_TYPE_WRAPPER_GET(CubicBezierEasingParameters, value);

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::CubicBezierEasingParameters, 1)
