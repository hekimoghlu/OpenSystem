/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include "UIGamepadProvider.h"

#if ENABLE(GAMEPAD) && OS(LINUX)

#include "WebPageProxy.h"
#include <WebCore/ManetteGamepadProvider.h>
#include <wtf/ProcessPrivilege.h>

namespace WebKit {

using namespace WebCore;

void UIGamepadProvider::platformSetDefaultGamepadProvider()
{
    if (GamepadProvider::singleton().isMockGamepadProvider())
        return;

    GamepadProvider::setSharedProvider(ManetteGamepadProvider::singleton());
}

#if !PLATFORM(GTK)
WebPageProxy* UIGamepadProvider::platformWebPageProxyForGamepadInput()
{
    return nullptr;
}
#endif

void UIGamepadProvider::platformStopMonitoringInput()
{
}

void UIGamepadProvider::platformStartMonitoringInput()
{
}

}

#endif // ENABLE(GAMEPAD) && OS(LINUX)
