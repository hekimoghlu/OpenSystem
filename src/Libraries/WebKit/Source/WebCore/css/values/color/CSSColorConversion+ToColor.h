/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

#include "CSSColorDescriptors.h"
#include "Color.h"
#include "ColorTypes.h"

namespace WebCore {

// This file implements support for converting "typed colors" (e.g. `SRGBA<float>`)
// into the type erased `Color`, allowing for compaction and optional flags.

constexpr bool outsideSRGBGamut(HSLA<float> hsla)
{
    auto unresolved = hsla.unresolved();
    return unresolved.saturation > 100.0 || unresolved.lightness < 0.0 || unresolved.lightness > 100.0;
}

constexpr bool outsideSRGBGamut(HWBA<float> hwba)
{
    auto unresolved = hwba.unresolved();
    return unresolved.whiteness < 0.0 || unresolved.whiteness > 100.0 || unresolved.blackness < 0.0 || unresolved.blackness > 100.0;
}

constexpr bool outsideSRGBGamut(SRGBA<float>)
{
    return false;
}

template<typename Descriptor, CSSColorFunctionForm Form>
struct ConvertToColor;

template<typename Descriptor>
struct ConvertToColor<Descriptor, CSSColorFunctionForm::Absolute> {
    static Color convertToColor(GetColorType<Descriptor> color, unsigned nestingLevel)
    {
        if constexpr (Descriptor::allowConversionTo8BitSRGB) {
            if constexpr (Descriptor::syntax == CSSColorFunctionSyntax::Modern) {
                if (color.unresolved().anyComponentIsNone()) {
                    // If any component uses "none", we store the value as is to allow for storage of the special value as NaN.
                    return { color, Descriptor::flagsForAbsolute };
                }
            }

            if (outsideSRGBGamut(color)) {
                // If any component is outside the reference range, we store the value as is to allow for non-SRGB gamut values.
                return { color, Descriptor::flagsForAbsolute };
            }

            if (nestingLevel > 1) {
                // If the color is being consumed as part of a composition (relative color, color-mix, light-dark, etc.), we
                // store the value as is to allow for maximum precision.
                return { color, Descriptor::flagsForAbsolute };
            }

            // The explicit conversion to SRGBA<uint8_t> is an intentional performance optimization that allows storing the
            // color with no extra allocation for an extended color object. This is permissible in some case due to the
            // historical requirement that some syntaxes serialize using the legacy color syntax (rgb()/rgba()) and
            // historically have used the 8-bit rgba internal representation in engines.
            return { convertColor<SRGBA<uint8_t>>(color), Descriptor::flagsForAbsolute };
        } else
            return { color, Descriptor::flagsForAbsolute };
    }
};

template<typename Descriptor>
struct ConvertToColor<Descriptor, CSSColorFunctionForm::Relative> {
    static Color convertToColor(GetColorType<Descriptor> color)
    {
        return { color, Descriptor::flagsForRelative };
    }
};

template<typename Descriptor, CSSColorFunctionForm Form>
Color convertToColor(GetColorType<Descriptor> color, unsigned nestingLevel)
{
    static_assert(Form == CSSColorFunctionForm::Absolute);
    return ConvertToColor<Descriptor, Form>::convertToColor(color, nestingLevel);
}

template<typename Descriptor, CSSColorFunctionForm Form>
Color convertToColor(GetColorType<Descriptor> color)
{
    static_assert(Form == CSSColorFunctionForm::Relative);
    return ConvertToColor<Descriptor, Form>::convertToColor(color);
}

template<typename Descriptor, CSSColorFunctionForm Form>
Color convertToColor(std::optional<GetColorType<Descriptor>> color, unsigned nestingLevel)
{
    static_assert(Form == CSSColorFunctionForm::Absolute);

    if (!color)
        return { };
    return convertToColor<Descriptor, Form>(*color, nestingLevel);
}

template<typename Descriptor, CSSColorFunctionForm Form>
Color convertToColor(std::optional<GetColorType<Descriptor>> color)
{
    static_assert(Form == CSSColorFunctionForm::Relative);
    if (!color)
        return { };
    return convertToColor<Descriptor, Form>(*color);
}

} // namespace WebCore
