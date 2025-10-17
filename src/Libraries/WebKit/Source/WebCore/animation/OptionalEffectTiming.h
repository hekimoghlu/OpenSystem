/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

#include "FillMode.h"
#include "PlaybackDirection.h"
#include "WebAnimationTypes.h"
#include <variant>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct OptionalEffectTiming {
    std::optional<std::variant<double, String>> duration;
    std::optional<double> iterations; // This value cannot be a MarkableDouble since we need to check for a NaN value.
    MarkableDouble delay;
    MarkableDouble endDelay;
    MarkableDouble iterationStart;
    String easing;
    OptionalFillMode fill;
    OptionalPlaybackDirection direction;
};

} // namespace WebCore
