/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

struct NoConversionDataRequiredToken;

namespace CSS {

// MARK: Angle

double canonicalizeAngle(double value, AngleUnit);

template<auto R, typename V> double canonicalize(AngleRaw<R, V> raw)
{
    return canonicalizeAngle(raw.value, raw.unit);
}

// MARK: Time

double canonicalizeTime(double, TimeUnit);

template<auto R, typename V> double canonicalize(TimeRaw<R, V> raw)
{
    return canonicalizeTime(raw.value, raw.unit);
}

// MARK: Frequency

double canonicalizeFrequency(double, FrequencyUnit);

template<auto R, typename V> double canonicalize(FrequencyRaw<R, V> raw)
{
    return canonicalizeFrequency(raw.value, raw.unit);
}

// MARK: Resolution

double canonicalizeResolution(double, ResolutionUnit);

template<auto R, typename V> double canonicalize(ResolutionRaw<R, V> raw)
{
    return canonicalizeResolution(raw.value, raw.unit);
}

} // namespace CSS
} // namespace WebCore
