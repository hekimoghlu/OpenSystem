/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include "ScreenManager.h"

#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
#include "WebProcessMessages.h"
#include "WebProcessPool.h"

namespace WebKit {

ScreenManager& ScreenManager::singleton()
{
    static NeverDestroyed<ScreenManager> manager;
    return manager;
}

PlatformDisplayID ScreenManager::displayID(PlatformScreen* screen) const
{
    return m_screenToDisplayIDMap.get(screen);
}

PlatformScreen* ScreenManager::screen(PlatformDisplayID displayID) const
{
    for (const auto& iter : m_screenToDisplayIDMap) {
        if (iter.value == displayID)
            return iter.key;
    }
    return nullptr;
}

void ScreenManager::addScreen(PlatformScreen* screen)
{
    m_screens.append(screen);
    m_screenToDisplayIDMap.add(screen, generatePlatformDisplayID(screen));
}

void ScreenManager::removeScreen(PlatformScreen* screen)
{
    m_screenToDisplayIDMap.remove(screen);
    m_screens.removeFirstMatching([screen](const auto& item) {
        return item.get() == screen;
    });
}

void ScreenManager::propertiesDidChange() const
{
    auto properties = collectScreenProperties();
    for (auto& pool : WebProcessPool::allProcessPools())
        pool->sendToAllProcesses(Messages::WebProcess::SetScreenProperties(properties));
}

} // namespace WebKit

#endif // PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
