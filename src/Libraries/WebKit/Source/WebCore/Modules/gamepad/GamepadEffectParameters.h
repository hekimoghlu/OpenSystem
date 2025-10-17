/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

#if ENABLE(GAMEPAD)

#include <wtf/Seconds.h>

namespace WebCore {

struct GamepadEffectParameters {
    double duration = 0.0;
    double startDelay = 0.0;
    double strongMagnitude = 0.0;
    double weakMagnitude = 0.0;

    double leftTrigger = 0.0;
    double rightTrigger = 0.0;

    // A maximum duration of 5 seconds is recommended by the specification:
    // - https://w3c.github.io/gamepad/extensions.html#gamepadeffectparameters-dictionary
    static constexpr Seconds maximumDuration = 5_s;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
