/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#include "CSSColor.h"
#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// hue-rotate() = hue-rotate( [ <angle> | <zero> ]? )
// https://drafts.fxtf.org/filter-effects/#funcdef-filter-hue-rotate
struct HueRotate {
    using Parameter = Angle<>;

    Markable<Parameter> value;

    bool operator==(const HueRotate&) const = default;
};
using HueRotateFunction = FunctionNotation<CSSValueHueRotate, HueRotate>;

DEFINE_TYPE_WRAPPER_GET(HueRotate, value);

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::HueRotate, 1)
