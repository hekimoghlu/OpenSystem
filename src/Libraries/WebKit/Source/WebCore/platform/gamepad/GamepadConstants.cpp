/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include "config.h"
#include "GamepadConstants.h"

#if ENABLE(GAMEPAD)

#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

const GamepadButtonRole maximumGamepadButton = GamepadButtonRole::CenterClusterCenter;
const size_t numberOfStandardGamepadButtonsWithoutHomeButton = static_cast<size_t>(maximumGamepadButton);
const size_t numberOfStandardGamepadButtonsWithHomeButton = numberOfStandardGamepadButtonsWithoutHomeButton + 1;

const String& standardGamepadMappingString()
{
    static NeverDestroyed<String> standardGamepadMapping = "standard"_s;
    return standardGamepadMapping;
}

#if ENABLE(WEBXR)
// https://immersive-web.github.io/webxr-gamepads-module/#dom-gamepadmappingtype-xr-standard
const String& xrStandardGamepadMappingString()
{
    static NeverDestroyed<String> xrStandardGamepadMapping = "xr-standard"_s;
    return xrStandardGamepadMapping;
}
#endif


} // namespace WebCore

#endif // ENABLE(GAMEPAD)
