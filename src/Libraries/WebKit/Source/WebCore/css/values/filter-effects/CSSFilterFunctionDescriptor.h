/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
#include "CSSValueKeywords.h"
#include "Color.h"
#include "FilterOperation.h"
#include "StyleCurrentColor.h"

namespace WebCore {

template<auto FilterFunction> struct CSSFilterFunctionDescriptor;

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-blur
template<> struct CSSFilterFunctionDescriptor<CSSValueBlur> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = false;

    static constexpr bool allowsValuesGreaterThanOne = false;
    static constexpr auto defaultValue = CSS::LengthRaw<> { CSS::LengthUnit::Px, 0 };
    static constexpr auto initialLengthValueForInterpolation = CSS::LengthRaw<> { CSS::LengthUnit::Px, 0 };

    static constexpr auto operationType = FilterOperation::Type::Blur;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-brightness
template<> struct CSSFilterFunctionDescriptor<CSSValueBrightness> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = true;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 1 };

    static constexpr auto operationType = FilterOperation::Type::Brightness;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-contrast
template<> struct CSSFilterFunctionDescriptor<CSSValueContrast> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = true;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 1 };

    static constexpr auto operationType = FilterOperation::Type::Contrast;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-drop-shadow
template<> struct CSSFilterFunctionDescriptor<CSSValueDropShadow> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = false;

    static constexpr auto defaultColorValue = Color::transparentBlack; // FIXME: This should be "currentcolor", but that requires filters to be able to store StyleColors.
    static constexpr auto defaultStdDeviationValue = CSS::LengthRaw<> { CSS::LengthUnit::Px, 0 };

    static constexpr auto initialColorValueForInterpolation = Color::transparentBlack;
    static constexpr auto initialLengthValueForInterpolation = CSS::LengthRaw<> { CSS::LengthUnit::Px, 0 };

    static constexpr auto operationType = FilterOperation::Type::DropShadow;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-grayscale
template<> struct CSSFilterFunctionDescriptor<CSSValueGrayscale> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = false;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 0 };

    static constexpr auto operationType = FilterOperation::Type::Grayscale;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-hue-rotate
template<> struct CSSFilterFunctionDescriptor<CSSValueHueRotate> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr auto defaultValue = CSS::AngleRaw<> { CSS::AngleUnit::Deg, 0 };
    static constexpr auto initialValueForInterpolation = CSS::AngleRaw<> { CSS::AngleUnit::Deg, 0 };

    static constexpr auto operationType = FilterOperation::Type::HueRotate;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-invert
template<> struct CSSFilterFunctionDescriptor<CSSValueInvert> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = false;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 0 };

    static constexpr auto operationType = FilterOperation::Type::Invert;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-opacity
template<> struct CSSFilterFunctionDescriptor<CSSValueOpacity> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = false;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 1 };

    static constexpr auto operationType = FilterOperation::Type::Opacity;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-saturate
template<> struct CSSFilterFunctionDescriptor<CSSValueSaturate> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = true;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 1 };

    static constexpr auto operationType = FilterOperation::Type::Saturate;
};

// https://drafts.fxtf.org/filter-effects/#funcdef-filter-sepia
template<> struct CSSFilterFunctionDescriptor<CSSValueSepia> {
    static constexpr bool isPixelFilterFunction = true;
    static constexpr bool isColorFilterFunction = true;

    static constexpr bool allowsValuesGreaterThanOne = false;
    static constexpr auto defaultValue = CSS::NumberRaw<> { 1 };
    static constexpr auto initialValueForInterpolation = CSS::NumberRaw<> { 0 };

    static constexpr auto operationType = FilterOperation::Type::Sepia;
};

// Non-standard addition.
template<> struct CSSFilterFunctionDescriptor<CSSValueAppleInvertLightness> {
    static constexpr bool isPixelFilterFunction = false;
    static constexpr bool isColorFilterFunction = true;

    static constexpr auto operationType = FilterOperation::Type::AppleInvertLightness;
};

template<auto filterFunction> static constexpr bool isPixelFilterFunction()
{
    return CSSFilterFunctionDescriptor<filterFunction>::isPixelFilterFunction;
}

template<auto filterFunction> static constexpr bool isColorFilterFunction()
{
    return CSSFilterFunctionDescriptor<filterFunction>::isColorFilterFunction;
}

template<auto filterFunction> static constexpr bool filterFunctionAllowsValuesGreaterThanOne()
{
    return CSSFilterFunctionDescriptor<filterFunction>::allowsValuesGreaterThanOne;
}

template<auto filterFunction> static constexpr decltype(auto) filterFunctionDefaultValue()
{
    return CSSFilterFunctionDescriptor<filterFunction>::defaultValue;
}

template<auto filterFunction> static constexpr decltype(auto) filterFunctionInitialValueForInterpolation()
{
    return CSSFilterFunctionDescriptor<filterFunction>::initialValueForInterpolation;
}

template<auto filterFunction> static constexpr decltype(auto) filterFunctionOperationType()
{
    return CSSFilterFunctionDescriptor<filterFunction>::operationType;
}

} // namespace WebCore
