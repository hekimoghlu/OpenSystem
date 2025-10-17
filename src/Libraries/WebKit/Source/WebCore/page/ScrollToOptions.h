/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

#include "ScrollOptions.h"
#include <cmath>

namespace WebCore {

struct ScrollToOptions : ScrollOptions {
    std::optional<double> left;
    std::optional<double> top;

    ScrollToOptions() = default;
    ScrollToOptions(double x, double y)
        : left(x)
        , top(y)
    { }
};

inline double normalizeNonFiniteValueOrFallBackTo(std::optional<double> value, double fallbackValue)
{
    // Normalize non-finite values (https://drafts.csswg.org/cssom-view/#normalize-non-finite-values).
    return value ? (std::isfinite(*value) ? *value : 0) : fallbackValue;
}

// FIXME(https://webkit.org/b/88339): Consider using FloatPoint or DoublePoint for fallback and return values.
inline ScrollToOptions normalizeNonFiniteCoordinatesOrFallBackTo(const ScrollToOptions& value, double x, double y)
{
    ScrollToOptions options = value;
    options.left = normalizeNonFiniteValueOrFallBackTo(value.left, x);
    options.top = normalizeNonFiniteValueOrFallBackTo(value.top, y);
    return options;
}

}
