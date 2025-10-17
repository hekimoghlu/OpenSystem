/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "HardwareAccelerationManager.h"

#include "AcceleratedBackingStore.h"

namespace WebKit {
using namespace WebCore;

HardwareAccelerationManager& HardwareAccelerationManager::singleton()
{
    static NeverDestroyed<HardwareAccelerationManager> manager;
    return manager;
}

HardwareAccelerationManager::HardwareAccelerationManager()
    : m_canUseHardwareAcceleration(true)
    , m_forceHardwareAcceleration(true)
{
#if !ENABLE(WEBGL)
    m_canUseHardwareAcceleration = false;
#else
    const char* disableCompositing = getenv("WEBKIT_DISABLE_COMPOSITING_MODE");
    if ((disableCompositing && strcmp(disableCompositing, "0")) || !AcceleratedBackingStore::checkRequirements())
        m_canUseHardwareAcceleration = false;
#endif

    const char* forceCompositing = getenv("WEBKIT_FORCE_COMPOSITING_MODE");
    if (forceCompositing && !strcmp(forceCompositing, "0"))
        m_forceHardwareAcceleration = false;
}

} // namespace WebKit
