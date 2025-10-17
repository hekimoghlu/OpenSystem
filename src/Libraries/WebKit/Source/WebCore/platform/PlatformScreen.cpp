/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
#include "PlatformScreen.h"

#if PLATFORM(COCOA) || PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))

#include "ScreenProperties.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static ScreenProperties& screenProperties()
{
    static NeverDestroyed<ScreenProperties> screenProperties;
    return screenProperties;
}

const ScreenProperties& getScreenProperties()
{
    return screenProperties();
}

PlatformDisplayID primaryScreenDisplayID()
{
    return screenProperties().primaryDisplayID;
}

void setScreenProperties(const ScreenProperties& properties)
{
    screenProperties() = properties;
}

const ScreenData* screenData(PlatformDisplayID screenDisplayID)
{
    if (screenProperties().screenDataMap.isEmpty())
        return nullptr;

    // Return property of the first screen if the screen is not found in the map.
    if (auto displayID = screenDisplayID ? screenDisplayID : primaryScreenDisplayID()) {
        auto properties = screenProperties().screenDataMap.find(displayID);
        if (properties != screenProperties().screenDataMap.end())
            return &properties->value;
    }

    // Last resort: use the first item in the screen list.
    return &screenProperties().screenDataMap.begin()->value;
}

} // namespace WebCore

#endif // PLATFORM(COCOA) || PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
