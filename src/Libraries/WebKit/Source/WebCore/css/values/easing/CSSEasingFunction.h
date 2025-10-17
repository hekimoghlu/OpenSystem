/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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

#include "CSSCubicBezierEasingFunction.h"
#include "CSSLinearEasingFunction.h"
#include "CSSSpringEasingFunction.h"
#include "CSSStepsEasingFunction.h"

namespace WebCore {
namespace CSS {

// `EasingFunction` uses a wrapping a struct to allow forward declaration.
struct EasingFunction {
    std::variant<
        // <linear()>
        Keyword::Linear,        // Equivalent to linear(0, 1)
        LinearEasingFunction,

        // <cubic-bezier()>
        Keyword::Ease,          // Equivalent to cubic-bezier(0.25, 0.1, 0.25, 1)
        Keyword::EaseIn,        // Equivalent to cubic-bezier(0.42, 0, 1, 1)
        Keyword::EaseOut,       // Equivalent to cubic-bezier(0, 0, 0.58, 1)
        Keyword::EaseInOut,     // Equivalent to cubic-bezier(0.42, 0, 0.58, 1)
        CubicBezierEasingFunction,

        // <steps()>
        Keyword::StepStart,     // Equivalent to steps(1, start)
        Keyword::StepEnd,       // Equivalent to steps(1, end)
        StepsEasingFunction,

        // <spring()>
        SpringEasingFunction
    > value;

    template<typename... F> decltype(auto) switchOn(F&&... f) const
    {
        return WTF::switchOn(value, std::forward<F>(f)...);
    }

    bool operator==(const EasingFunction&) const = default;
};

} // namespace CSS
} // namespace WebCore

template<> inline constexpr auto WebCore::TreatAsVariantLike<WebCore::CSS::EasingFunction> = true;
