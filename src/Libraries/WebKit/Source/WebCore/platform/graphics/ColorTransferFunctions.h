/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include <wtf/MathExtras.h>

namespace WebCore {

// Transfer functions for colors that can be gamma encoded.

enum class TransferFunctionMode : uint8_t {
    Clamped,
    Unclamped
};

template<typename T, TransferFunctionMode mode>
struct A98RGBTransferFunction {
    static T toGammaEncoded(T);
    static T toLinear(T);
};

template<typename T, TransferFunctionMode mode>
struct ProPhotoRGBTransferFunction {
    static constexpr T gamma = 1.8;

    static T toGammaEncoded(T);
    static T toLinear(T);
};

template<typename T, TransferFunctionMode mode>
struct Rec2020TransferFunction {
    static constexpr T alpha = 1.09929682680944;
    static constexpr T beta = 0.018053968510807;
    static constexpr T gamma = 0.45;

    static T toGammaEncoded(T);
    static T toLinear(T);
};

template<typename T, TransferFunctionMode mode>
struct SRGBTransferFunction {
    static T toGammaEncoded(T);
    static T toLinear(T);
};

// MARK: A98RGBTransferFunction.

template<typename T, TransferFunctionMode mode> T A98RGBTransferFunction<T, mode>::toGammaEncoded(T c)
{
    auto sign = std::signbit(c) ? -1.0f : 1.0f;
    auto result = std::pow(std::abs(c), 256.0f / 563.0f) * sign;
    if constexpr (mode == TransferFunctionMode::Clamped)
        return clampTo<T>(result, 0, 1);
    return result;
}

template<typename T, TransferFunctionMode mode> T A98RGBTransferFunction<T, mode>::toLinear(T c)
{
    auto sign = std::signbit(c) ? -1.0f : 1.0f;
    auto result = std::pow(std::abs(c), 563.0f / 256.0f) * sign;
    if constexpr (mode == TransferFunctionMode::Clamped)
        return clampTo<T>(result, 0, 1);
    return result;
}

// MARK: ProPhotoRGBTransferFunction.

template<typename T, TransferFunctionMode mode> T ProPhotoRGBTransferFunction<T, mode>::toGammaEncoded(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c < 1.0 / 512.0)
            return 16.0 * c;

        return clampTo<T>(std::pow(c, 1.0 / gamma), 0, 1);
    } else {
        if (std::abs(c) < 1.0 / 512.0)
            return 16.0 * c;

        float sign = std::signbit(c) ? -1.0 : 1.0;
        return std::pow(c, 1.0 / gamma) * sign;
    }
}

template<typename T, TransferFunctionMode mode> T ProPhotoRGBTransferFunction<T, mode>::toLinear(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c <= 16.0 / 512.0)
            return c / 16.0;

        return clampTo<T>(std::pow(c, gamma), 0, 1);
    } else {
        if (std::abs(c) <= 16.0 / 512.0)
            return c / 16.0;

        float sign = std::signbit(c) ? -1.0 : 1.0;
        return std::pow(c, gamma) * sign;
    }
}

// MARK: Rec2020TransferFunction.

template<typename T, TransferFunctionMode mode> T Rec2020TransferFunction<T, mode>::toGammaEncoded(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c <= beta)
            return 4.5f * c;

        return clampTo<T>(alpha * std::pow(c, gamma) - (alpha - 1.0f), 0, 1);
    } else {
        if (std::abs(c) <= beta)
            return 4.5f * c;

        float sign = std::signbit(c) ? -1.0 : 1.0;
        return (alpha * std::pow(c, gamma) - (alpha - 1.0)) * sign;
    }
}

template<typename T, TransferFunctionMode mode> T Rec2020TransferFunction<T, mode>::toLinear(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c < beta * 4.5f)
            return c / 4.5f;
        return clampTo<T>(std::pow((c + alpha - 1.0) / alpha, 1.0 / gamma), 0, 1);
    } else {
        if (std::abs(c) < beta * 4.5f)
            return c / 4.5f;

        float sign = std::signbit(c) ? -1.0 : 1.0;
        return std::pow((c + alpha - 1.0) / alpha, 1.0 / gamma) * sign;
    }
}

// MARK: SRGBTransferFunction.

template<typename T, TransferFunctionMode mode> T SRGBTransferFunction<T, mode>::toGammaEncoded(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c < 0.0031308f)
            return std::max<T>(12.92f * c, 0);
        return clampTo<T>(1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f, 0, 1);
    } else {
        auto sign = std::signbit(c) ? -1.0f : 1.0f;
        c = std::abs(c);

        if (c < 0.0031308f)
            return 12.92f * c * sign;
        return (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f) * sign;
    }
}

template<typename T, TransferFunctionMode mode> T SRGBTransferFunction<T, mode>::toLinear(T c)
{
    if constexpr (mode == TransferFunctionMode::Clamped) {
        if (c <= 0.04045f)
            return std::max<float>(c / 12.92f, 0);
        return clampTo<float>(std::pow((c + 0.055f) / 1.055f, 2.4f), 0, 1);
    } else {
        auto sign = std::signbit(c) ? -1.0f : 1.0f;
        c = std::abs(c);

        if (c <= 0.04045f)
            return c / 12.92f * sign;
        return std::pow((c + 0.055f) / 1.055f, 2.4f) * sign;
    }
}

}
