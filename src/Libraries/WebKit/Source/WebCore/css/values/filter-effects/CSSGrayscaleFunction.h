/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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

// grayscale() = grayscale( [ <number [0,1(clamp upper)] > | <percentage [0,100(clamp upper)]> ]? )
// https://drafts.fxtf.org/filter-effects/#funcdef-filter-grayscale
struct Grayscale {
    using Parameter = NumberOrPercentage<ClosedUnitRangeClampUpper, ClosedPercentageRangeClampUpper>;

    Markable<Parameter> value;

    bool operator==(const Grayscale&) const = default;
};
using GrayscaleFunction = FunctionNotation<CSSValueGrayscale, Grayscale>;

DEFINE_TYPE_WRAPPER_GET(Grayscale, value);

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Grayscale, 1)
