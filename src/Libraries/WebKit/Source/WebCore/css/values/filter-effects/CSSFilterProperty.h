/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#include "CSSBlurFunction.h"
#include "CSSBrightnessFunction.h"
#include "CSSContrastFunction.h"
#include "CSSDropShadowFunction.h"
#include "CSSFilterReference.h"
#include "CSSGrayscaleFunction.h"
#include "CSSHueRotateFunction.h"
#include "CSSInvertFunction.h"
#include "CSSOpacityFunction.h"
#include "CSSSaturateFunction.h"
#include "CSSSepiaFunction.h"

namespace WebCore {
namespace CSS {

using Filter = std::variant<
    BlurFunction,
    BrightnessFunction,
    ContrastFunction,
    DropShadowFunction,
    GrayscaleFunction,
    HueRotateFunction,
    InvertFunction,
    OpacityFunction,
    SaturateFunction,
    SepiaFunction,
    FilterReference
>;
using FilterValueList = SpaceSeparatedVector<Filter>;

// <'filter'> = none | <filter-value-list>
// https://drafts.fxtf.org/filter-effects/#propdef-filter
// NOTE: Subclassing, rather than aliasing, is being used to allow easy forward declarations.
struct FilterProperty : ListOrNone<FilterValueList> { using ListOrNone<FilterValueList>::ListOrNone; };

} // namespace CSS
} // namespace WebCore

template<> inline constexpr auto WebCore::TreatAsVariantLike<WebCore::CSS::FilterProperty> = true;
