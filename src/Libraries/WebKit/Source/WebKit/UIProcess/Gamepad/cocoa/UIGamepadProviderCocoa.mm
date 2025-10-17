/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#import "config.h"
#import "UIGamepadProvider.h"

#if ENABLE(GAMEPAD)

#import <WebCore/GameControllerGamepadProvider.h>
#import <WebCore/HIDGamepadProvider.h>
#import <WebCore/MockGamepadProvider.h>
#import <WebCore/MultiGamepadProvider.h>

namespace WebKit {
using namespace WebCore;

#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
static bool useGameControllerFramework = true;
#else
static bool useGameControllerFramework = false;
#endif

void UIGamepadProvider::setUsesGameControllerFramework()
{
    useGameControllerFramework = true;
}

void UIGamepadProvider::platformSetDefaultGamepadProvider()
{
    if (GamepadProvider::singleton().isMockGamepadProvider())
        return;

#if PLATFORM(IOS_FAMILY)
    GamepadProvider::setSharedProvider(GameControllerGamepadProvider::singleton());
#else
    if (useGameControllerFramework)
        GamepadProvider::setSharedProvider(GameControllerGamepadProvider::singleton());
    else {
#if HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
        GamepadProvider::setSharedProvider(MultiGamepadProvider::singleton());
#else
        GamepadProvider::setSharedProvider(HIDGamepadProvider::singleton());
#endif // HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
    }
#endif // PLATFORM(IOS_FAMILY)
}

void UIGamepadProvider::platformStopMonitoringInput()
{
#if PLATFORM(MAC)
    if (!useGameControllerFramework)
        HIDGamepadProvider::singleton().stopMonitoringInput();
#endif
}

void UIGamepadProvider::platformStartMonitoringInput()
{
#if PLATFORM(MAC)
    if (!useGameControllerFramework)
        HIDGamepadProvider::singleton().startMonitoringInput();
#endif
}

}

#endif // ENABLE(GAMEPAD)
