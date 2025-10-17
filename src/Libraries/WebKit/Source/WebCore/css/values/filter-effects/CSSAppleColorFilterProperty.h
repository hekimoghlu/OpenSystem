/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

#include "CSSAppleInvertLightnessFunction.h"
#include "CSSBrightnessFunction.h"
#include "CSSContrastFunction.h"
#include "CSSGrayscaleFunction.h"
#include "CSSHueRotateFunction.h"
#include "CSSInvertFunction.h"
#include "CSSOpacityFunction.h"
#include "CSSSaturateFunction.h"
#include "CSSSepiaFunction.h"

namespace WebCore {
namespace CSS {

using AppleColorFilter = std::variant<
    AppleInvertLightnessFunction,
    BrightnessFunction,
    ContrastFunction,
    GrayscaleFunction,
    HueRotateFunction,
    InvertFunction,
    OpacityFunction,
    SaturateFunction,
    SepiaFunction
>;
using AppleColorFilterValueList = SpaceSeparatedVector<AppleColorFilter>;

// Non-standard type used for the `-apple-color-filter` property. It is similar to CSS::FilterProperty,
// but does not support `blur()`, `drop-shadow()` and reference filters, but adds support for the
// non-standard function `-apple-invert-lightness-filter()`.
// <'-apple-color-filter'> = none | <-apple-color-filter-value-list>
// NOTE: Subclassing, rather than aliasing, is being used to allow easy forward declarations.
struct AppleColorFilterProperty : ListOrNone<AppleColorFilterValueList> { using ListOrNone<AppleColorFilterValueList>::ListOrNone; };

} // namespace CSS
} // namespace WebCore

template<> inline constexpr auto WebCore::TreatAsVariantLike<WebCore::CSS::AppleColorFilterProperty> = true;
