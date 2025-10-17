/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

#include <wtf/Forward.h>

namespace WebCore {

// Buttons in the "standard" gamepad layout in the Web Gamepad spec
// https://www.w3.org/TR/gamepad/#dfn-standard-gamepad-layout
enum class GamepadButtonRole : uint8_t {
    RightClusterBottom = 0,
    RightClusterRight = 1,
    RightClusterLeft = 2,
    RightClusterTop = 3,
    LeftShoulderFront = 4,
    RightShoulderFront = 5,
    LeftShoulderBack = 6,
    RightShoulderBack = 7,
    CenterClusterLeft = 8,
    CenterClusterRight = 9,
    LeftStick = 10,
    RightStick = 11,
    LeftClusterTop = 12,
    LeftClusterBottom = 13,
    LeftClusterLeft = 14,
    LeftClusterRight = 15,
    CenterClusterCenter = 16,
    Nonstandard1 = 17,
    Nonstandard2 = 18,
};

extern const size_t numberOfStandardGamepadButtonsWithoutHomeButton;
extern const size_t numberOfStandardGamepadButtonsWithHomeButton;
extern const GamepadButtonRole maximumGamepadButton;

const String& standardGamepadMappingString();
#if ENABLE(WEBXR)
const String& xrStandardGamepadMappingString();
#endif

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
